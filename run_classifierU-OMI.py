import os
import random
import argparse
import time
import hashlib
import json
from uer.layers.layer_norm import LayerNorm, T5LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score, confusion_matrix, precision_score, recall_score,
    roc_auc_score, roc_curve, auc as calc_auc
)

from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts



def wall_time(f):

    def wrap(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = f(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        return out, dt
    return wrap


def md5(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except Exception:
        return None


def safe_torch_load(path, map_location="cpu"):

    kw = {"map_location": map_location}
    try:
        kw["weights_only"] = True
    except TypeError:
        pass
    obj = torch.load(path, **kw)
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
    if isinstance(obj, (dict, nn.Module)) or isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint format for {path}")



class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha


        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)


        self.exit_layers = getattr(args, "exit_layers", [])
        self.enable_early_exit = getattr(args, "enable_early_exit", False)
        self.early_exit_metric = getattr(args, "early_exit_metric", "mutual_info")
        self.early_exit_threshold = float(getattr(args, "early_exit_threshold", 0.0))
        self.train_exits = getattr(args, "train_exits", False)
        self.min_exit_layer = int(getattr(args, "min_exit_layer", 1))
        self.normalize_collected = bool(getattr(args, "normalize_collected", True))


        self.early_exit_low = float(getattr(args, "early_exit_low",
                                            self.early_exit_threshold))

        self.early_exit_high = getattr(args, "early_exit_high", None)
        if self.early_exit_high is not None:
            self.early_exit_high = float(self.early_exit_high)

        self.per_layer_high = getattr(args, "per_layer_high", None)
        self.per_layer_low = getattr(args, "per_layer_low", None)


        self.osr_mode = False


        self.exit_classifiers = nn.ModuleList(
            [nn.Linear(args.hidden_size, self.labels_num) for _ in self.exit_layers]
        )


        self.collect_norm = None
        if self.normalize_collected and getattr(args, "layernorm_positioning", "pre") == "pre":
            enc_ln = getattr(self.encoder, "layer_norm", None)
            if isinstance(enc_ln, nn.Module):
                self.collect_norm = enc_ln
            else:
                ln_cls = T5LayerNorm if getattr(args, "layernorm", "bert") == "t5" else LayerNorm
                self.collect_norm = ln_cls(args.hidden_size)


        self.total_layers = int(getattr(args, "layers_num", len(self.exit_layers) + 1))


    def _pool(self, hidden):
        if self.pooling == "mean":
            return torch.mean(hidden, dim=1)
        elif self.pooling == "max":
            return torch.max(hidden, dim=1)[0]
        elif self.pooling == "last":
            return hidden[:, -1, :]
        else:
            return hidden[:, 0, :]

    def _maybe_norm(self, h):
        return self.collect_norm(h) if self.collect_norm is not None else h

    def _logits_from_hidden(self, hidden, head_linear):
        hidden = self._maybe_norm(hidden)
        pooled = self._pool(hidden)
        proj = torch.tanh(self.output_layer_1(pooled))
        logits = head_linear(proj)
        return logits


    def reverse_kl(self, alpha, beta):
        alpha_cpu = alpha.to("cpu")
        beta_cpu = beta.to("cpu")
        alpha0_cpu = torch.sum(alpha_cpu, dim=1, keepdim=True)
        beta0_cpu = torch.sum(beta_cpu, dim=1, keepdim=True)

        t1 = torch.lgamma(alpha0_cpu) - torch.sum(torch.lgamma(alpha_cpu), dim=1, keepdim=True)
        t2 = torch.sum((alpha_cpu - beta_cpu) * (torch.digamma(alpha_cpu) - torch.digamma(alpha0_cpu)), dim=1, keepdim=True)
        t3 = -torch.lgamma(beta0_cpu) + torch.sum(torch.lgamma(beta_cpu), dim=1, keepdim=True)
        kl = (t1 + t2 + t3).to(alpha.device)
        return torch.mean(kl.squeeze())

    def build_beta(self, targets, num_classes, beta_val=100.0):
        beta = torch.ones((targets.size(0), num_classes), device=targets.device)
        known_mask = (targets != (num_classes - 1))
        if known_mask.sum() > 0:
            beta[known_mask, :] = 1.0
            beta[known_mask, targets[known_mask]] = beta_val
        return beta

    def compute_uncertainty(self, alphas, alpha0):
        probs = alphas / (alpha0 + 1e-12)
        entropy_of_expected = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        expected_entropy = self.expected_entropy_from_alphas(alphas, alpha0)
        mutual_info = entropy_of_expected - expected_entropy
        K = alphas.size(1)
        k_over_alpha0 = K / (alpha0 + 1e-12)
        return {
            "mutual_info": mutual_info,
            "expected_entropy": expected_entropy,
            "entropy_of_expected": entropy_of_expected,
            "k_over_alpha0": k_over_alpha0.squeeze(-1),
        }

    def expected_entropy_from_alphas(self, alphas, alpha0):
        if alpha0 is None:
            alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        alphas_cpu = alphas.detach().cpu()
        alpha0_cpu = alpha0.detach().cpu()
        expected_entropy = -torch.sum(
            (alphas_cpu / (alpha0_cpu + 1e-12)) *
            (torch.digamma(alphas_cpu + 1) - torch.digamma(alpha0_cpu + 1)),
            dim=1
        )
        return expected_entropy.to(alphas.device)


    def forward(self, src, tgt, seg, soft_tgt=None):

        emb = self.embedding(src, seg)


        if tgt is not None:
            final_hidden, collected = None, []
            if len(self.exit_layers) > 0:
                try:
                    final_hidden, collected = self.encoder(emb, seg, return_layers_idx=self.exit_layers)
                except TypeError:
                    final_hidden = self.encoder(emb, seg)
                    collected = []
            else:
                final_hidden = self.encoder(emb, seg)

            exit_hiddens, exit_heads = [], []
            if len(collected) == len(self.exit_layers):
                for h, head in zip(collected, self.exit_classifiers):
                    exit_hiddens.append(h)
                    exit_heads.append(head)

            exit_hiddens.append(final_hidden)
            exit_heads.append(self.output_layer_2)

            losses = []
            last_logits, last_uncertainty = None, None
            for h, head in zip(exit_hiddens, exit_heads):
                logits = self._logits_from_hidden(h, head)
                alphas = torch.exp(logits.clamp(max=30))
                beta = self.build_beta(tgt, self.labels_num, beta_val=100.0).to(alphas.device)
                loss = self.reverse_kl(alphas, beta)
                losses.append(loss)

                last_logits = logits
                alpha0 = torch.sum(alphas, dim=1, keepdim=True)
                last_uncertainty = self.compute_uncertainty(alphas, alpha0)


            if self.train_exits and len(losses) > 1:
                final_loss = losses[-1]
                early_losses = torch.stack(losses[:-1])
                lam = getattr(self.args, "exit_loss_weight", 0.3)
                loss_out = final_loss + lam * early_losses.mean()
            else:
                loss_out = losses[-1]
            return loss_out, last_logits, last_uncertainty


        if self.enable_early_exit and len(self.exit_layers) > 0:

            head_map = {int(l): head for l, head in zip(self.exit_layers, self.exit_classifiers)}
            final_layer_id = self.total_layers if self.total_layers else (
                max(self.exit_layers) if self.exit_layers else 12
            )
            head_map[int(final_layer_id)] = self.output_layer_2


            def pick_metric(unc_dict):
                if self.early_exit_metric == "entropy_of_expected":
                    return unc_dict["entropy_of_expected"]
                elif self.early_exit_metric == "expected_entropy":
                    return unc_dict["expected_entropy"]
                elif self.early_exit_metric == "k_over_alpha0":
                    return unc_dict.get("k_over_alpha0", unc_dict["mutual_info"])
                else:
                    return unc_dict["mutual_info"]


            base_low_thr = float(getattr(self, "early_exit_low", self.early_exit_threshold))
            global_high_thr = getattr(self, "early_exit_high", None)


            per_layer_high = getattr(self.args, "per_layer_high", getattr(self, "per_layer_high", None))
            per_layer_low = getattr(self.args, "per_layer_low", getattr(self, "per_layer_low", None))


            osr_flag = bool(getattr(self, "osr_mode", getattr(self.args, "osr_mode", False)))


            def get_low_thr_for_layer(layer_id: int) -> float:
                if isinstance(per_layer_low, dict):
                    if str(layer_id) in per_layer_low:
                        return float(per_layer_low[str(layer_id)])
                    if int(layer_id) in per_layer_low:
                        return float(per_layer_low[int(layer_id)])
                return base_low_thr

            def get_high_thr_for_layer(layer_id: int):
                if isinstance(per_layer_high, dict):
                    if str(layer_id) in per_layer_high:
                        return float(per_layer_high[str(layer_id)])
                    if int(layer_id) in per_layer_high:
                        return float(per_layer_high[int(layer_id)])
                return None


            def decider_fn(layer_id: int, h_act: torch.Tensor) -> torch.Tensor:
                head = head_map.get(int(layer_id), self.output_layer_2)
                logits_i = self._logits_from_hidden(h_act, head)
                alphas_i = torch.exp(logits_i.clamp(max=30))
                alpha0_i = torch.sum(alphas_i, dim=1, keepdim=True)
                unc_i = self.compute_uncertainty(alphas_i, alpha0_i)
                m = pick_metric(unc_i)


                cur_low_thr = get_low_thr_for_layer(layer_id)
                low_exit = (m <= cur_low_thr)


                if not osr_flag:
                    return low_exit

                high_masks = []


                layer_high_thr = get_high_thr_for_layer(layer_id)
                if layer_high_thr is not None:
                    high_masks.append(m >= layer_high_thr)


                if global_high_thr is not None:
                    high_masks.append(m >= float(global_high_thr))

                if len(high_masks) == 0:
                    return low_exit

                high_exit = high_masks[0]
                for hm in high_masks[1:]:
                    high_exit = high_exit | hm


                return low_exit | high_exit


            chosen_hidden, exit_layer = self.encoder.forward_early_exit(
                emb=emb, seg=seg,
                exit_layers=self.exit_layers,
                min_exit_layer=self.min_exit_layer,
                decider_fn=decider_fn
            )


            chosen_hidden = self._maybe_norm(chosen_hidden)
            pooled = self._pool(chosen_hidden)
            proj = torch.tanh(self.output_layer_1(pooled))
            B = proj.size(0)
            C = self.labels_num
            chosen_logits = torch.empty((B, C), device=proj.device, dtype=proj.dtype)

            unique_layers = torch.unique(exit_layer)
            for lid in unique_layers.tolist():
                lid = int(lid)
                head = head_map.get(lid, self.output_layer_2)
                mask = (exit_layer == lid)
                if mask.any():
                    chosen_logits[mask] = head(proj[mask])


            alphas = torch.exp(chosen_logits.clamp(max=30))
            alpha0 = torch.sum(alphas, dim=1, keepdim=True)
            unc = self.compute_uncertainty(alphas, alpha0)
            uncertainty = {
                "exit_layer": exit_layer,
                "mutual_info": unc["mutual_info"],
                "entropy_of_expected": unc["entropy_of_expected"],
                "expected_entropy": unc["expected_entropy"],
                "k_over_alpha0": unc.get("k_over_alpha0"),
            }
            return None, chosen_logits, uncertainty

        final_hidden = self.encoder(emb, seg)
        logits = self._logits_from_hidden(final_hidden, self.output_layer_2)
        alphas = torch.exp(logits.clamp(max=30))
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        uncertainty = self.compute_uncertainty(alphas, alpha0)
        return None, logits, uncertainty




def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        state = safe_torch_load(args.pretrained_model_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("[CKPT][pretrained] missing:", missing, "unexpected:", unexpected)
    else:
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False
        )
    else:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters, lr=args.learning_rate,
            scale_parameter=False, relative_step=False
        )
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size: (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size:, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a)
                )
                seg = [1] * len(src)
            else:
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
                )
                src_b = args.tokenizer.convert_tokens_to_ids(
                    args.tokenizer.tokenize(text_b) + [SEP_TOKEN]
                )
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))
    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch,
                seg_batch, soft_tgt_batch=None):
    model.zero_grad()
    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()
    return loss


def evaluate(args, dataset, print_confusion_matrix=False):

    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size
    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    all_preds = []
    all_golds = []

    args.model.eval()
    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(
            batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits, _ = args.model(src_batch, tgt_batch, seg_batch)
        alphas = torch.exp(logits.clamp(max=30))
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / alpha0
        pred = torch.argmax(probs, dim=1)

        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

        all_preds.extend(pred.cpu().tolist())
        all_golds.extend(gold.cpu().tolist())

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        cf_array = confusion.numpy()
        os.makedirs("data2/results", exist_ok=True)
        with open("data2/results/matrixTatic.txt", 'w') as f:
            for cf_a in cf_array:
                f.write(str(cf_a) + '\n')
        print("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    eps = 1e-9
    acc = correct / (len(dataset) + eps)
    precision_macro = precision_score(all_golds, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_golds, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_golds, all_preds, average='macro', zero_division=0)

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(acc, correct, len(dataset)))
    print("Global Closed-set Metrics (train-style, no EE):")
    print("  Macro-Precision: {:.4f}".format(precision_macro))
    print("  Macro-Recall:    {:.4f}".format(recall_macro))
    print("  Macro-F1:        {:.4f}".format(f1_macro))

    return acc, confusion


def evaluate_infer(args, dataset, print_confusion_matrix=False):

    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size
    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    all_preds = []
    all_golds = []


    layer_correct = {}
    layer_total = {}
    layer_preds = {}
    layer_golds = {}

    args.model.eval()
    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(
            batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits, uncertainty = args.model(src_batch, tgt=None, seg=seg_batch)

        alphas = torch.exp(logits.clamp(max=30))
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / alpha0
        pred = torch.argmax(probs, dim=1)

        gold = tgt_batch


        exit_layer_batch = None
        if isinstance(uncertainty, dict) and "exit_layer" in uncertainty:
            el = uncertainty["exit_layer"]
            if torch.is_tensor(el):
                exit_layer_batch = el.detach().cpu().tolist()
            elif isinstance(el, (list, tuple, np.ndarray)):
                exit_layer_batch = list(el)
            else:
                exit_layer_batch = [int(el)] * pred.size(0)
        else:
            exit_layer_batch = [-1] * pred.size(0)

        for j in range(pred.size(0)):
            p = pred[j].item()
            g = gold[j].item()
            confusion[p, g] += 1
            correct += int(p == g)

            all_preds.append(p)
            all_golds.append(g)

            if exit_layer_batch is not None:
                lid = int(exit_layer_batch[j])
                layer_total[lid] = layer_total.get(lid, 0) + 1
                if p == g:
                    layer_correct[lid] = layer_correct.get(lid, 0) + 1
                if lid not in layer_preds:
                    layer_preds[lid] = []
                    layer_golds[lid] = []
                layer_preds[lid].append(p)
                layer_golds[lid].append(g)

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        cf_array = confusion.numpy()
        os.makedirs("data2/results", exist_ok=True)
        with open("data2/results/matrixTatic_EE.txt", 'w') as f:
            for cf_a in cf_array:
                f.write(str(cf_a) + '\n')
        print("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    eps = 1e-9
    acc = correct / (len(dataset) + eps)
    precision_macro = precision_score(all_golds, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_golds, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_golds, all_preds, average='macro', zero_division=0)

    model_inst = args.model.module if isinstance(args.model, torch.nn.DataParallel) else args.model
    ee_flag = bool(getattr(model_inst, "enable_early_exit", False))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(acc, correct, len(dataset)))
    print("Global Closed-set Metrics (infer-style, EE={}):".format(ee_flag))
    print("  Macro-Precision: {:.4f}".format(precision_macro))
    print("  Macro-Recall:    {:.4f}".format(recall_macro))
    print("  Macro-F1:        {:.4f}".format(f1_macro))


    if len(layer_total) > 0:
        total_samples = len(dataset)
        print("\nPer-exit-layer metrics (EE, closed-set on TEST):")
        for lid in sorted(layer_total.keys()):
            preds_l = np.array(layer_preds[lid])
            golds_l = np.array(layer_golds[lid])

            acc_l = np.mean(preds_l == golds_l)
            precision_l = precision_score(golds_l, preds_l, average='macro', zero_division=0)
            recall_l = recall_score(golds_l, preds_l, average='macro', zero_division=0)
            f1_l = f1_score(golds_l, preds_l, average='macro', zero_division=0)
            ratio_l = layer_total[lid] / (total_samples + 1e-9)

            correct_l = layer_correct.get(lid, 0)
            total_l = layer_total[lid]

            print(
                f"  Layer {lid:2d}: "
                f"ratio={ratio_l*100:6.2f}%, "
                f"Acc={acc_l:.4f}, "
                f"P={precision_l:.4f}, "
                f"R={recall_l:.4f}, "
                f"F1={f1_l:.4f} "
                f"(Correct/Total={correct_l}/{total_l})"
            )

    return acc, confusion


def test_1(args, dataset, device, tag="test", threshold=None):
    batch_size = args.batch_size
    UNKNOWN_LABEL = args.labels_num - 1
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    uncertainties, pred_labels, true_labels = [], [], []
    exit_layers = []

    args.model.eval()
    with torch.no_grad():
        for i in range(0, len(src), batch_size):
            src_batch = src[i:i+batch_size].to(device)
            seg_batch = seg[i:i+batch_size].to(device)
            _, logits, uncertainty = args.model(src_batch, tgt=None, seg=seg_batch)

            mi = uncertainty.get('mutual_info', None) if isinstance(uncertainty, dict) else None
            if mi is None:
                alphas_tmp = torch.exp(logits)
                alpha0_tmp = torch.sum(alphas_tmp, dim=1, keepdim=True)
                probs_tmp = alphas_tmp / (alpha0_tmp + 1e-12)
                entropy_of_expected = -torch.sum(
                    probs_tmp * torch.log(probs_tmp + 1e-10), dim=1
                )
                mi = entropy_of_expected
            u_batch = mi.detach().cpu().numpy() if torch.is_tensor(mi) else np.asarray(mi)

            alphas = torch.exp(logits)
            alpha0 = torch.sum(alphas, dim=1, keepdim=True)
            probs = (alphas / alpha0).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            exit_layer_batch = None
            if isinstance(uncertainty, dict) and "exit_layer" in uncertainty:
                el = uncertainty["exit_layer"]
                if torch.is_tensor(el):
                    exit_layer_batch = el.detach().cpu().tolist()
                elif isinstance(el, (list, tuple, np.ndarray)):
                    exit_layer_batch = list(el)
                else:
                    exit_layer_batch = [int(el)] * len(preds)
            else:
                exit_layer_batch = [-1] * len(preds)

            if threshold is not None:
                is_unknown = u_batch > threshold
                preds[is_unknown] = UNKNOWN_LABEL

            uncertainties.extend(u_batch)
            pred_labels.extend(preds)
            exit_layers.extend(exit_layer_batch)

            if tag == "open":
                true_labels.extend([UNKNOWN_LABEL]*len(preds))
            else:
                true_labels.extend(tgt[i:i+batch_size].cpu().numpy())

    return (
        np.array(uncertainties),
        np.array(pred_labels),
        np.array(true_labels),
        np.array(exit_layers)
    )


def evaluate_open_set_metrics(uncertainties, pred_labels, gt_labels,
                              threshold, known_label_num):

    UNKNOWN_CLASS = known_label_num - 1


    pred_labels_with_reject = np.where(
        uncertainties > threshold,
        UNKNOWN_CLASS,
        pred_labels
    )

    gt_labels = np.array(gt_labels)
    pred_labels_with_reject = np.array(pred_labels_with_reject)

    is_unknown      = (gt_labels == UNKNOWN_CLASS)
    is_pred_unknown = (pred_labels_with_reject == UNKNOWN_CLASS)

    y_true_bin = is_unknown.astype(int)
    y_pred_bin = is_pred_unknown.astype(int)

    if len(np.unique(y_true_bin)) < 2:

        open_acc = open_precision = open_recall = open_f1 = float('nan')
    else:
        open_acc = np.mean(y_true_bin == y_pred_bin)
        open_precision = precision_score(
            y_true_bin, y_pred_bin,
            average='binary', pos_label=1, zero_division=0
        )
        open_recall = recall_score(
            y_true_bin, y_pred_bin,
            average='binary', pos_label=1, zero_division=0
        )
        open_f1 = f1_score(
            y_true_bin, y_pred_bin,
            average='binary', pos_label=1, zero_division=0
        )


    closed_idx = (gt_labels != UNKNOWN_CLASS)
    filtered_true = gt_labels[closed_idx]
    filtered_pred = pred_labels_with_reject[closed_idx]

    if np.sum(closed_idx) > 0:

        labels_known = list(range(UNKNOWN_CLASS))

        precision_closed = precision_score(
            filtered_true, filtered_pred,
            average='macro', labels=labels_known, zero_division=0
        )
        recall_closed = recall_score(
            filtered_true, filtered_pred,
            average='macro', labels=labels_known, zero_division=0
        )
        f1_closed = f1_score(
            filtered_true, filtered_pred,
            average='macro', labels=labels_known, zero_division=0
        )
        acc_closed = np.mean(filtered_pred == filtered_true)
    else:
        precision_closed = recall_closed = f1_closed = acc_closed = float('nan')

    return {
        "Closed-set Acc": acc_closed,
        "Closed-set F1": f1_closed,
        "Closed-set Precision": precision_closed,
        "Closed-set Recall": recall_closed,
        "Open-set Acc": open_acc,
        "Open-set F1": open_f1,
        "Open-set Precision": open_precision,
        "Open-set Recall": open_recall,
    }



def _parse_exit_layers(s: str):
    if s is None or len(s.strip()) == 0:
        return []
    return [int(x) for x in s.split(",") if str(x).strip().isdigit()]


@wall_time
def timed_infer(args, dataset):

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    bs = args.batch_size
    n_tok = 0
    exit_layers = []

    args.model.eval()
    with torch.no_grad():
        for i in range(0, len(src), bs):
            src_batch = src[i:i+bs].to(args.device)
            seg_batch = seg[i:i+bs].to(args.device)
            _, logits, uncertainty = args.model(src_batch, tgt=None, seg=seg_batch)
            n_tok += src_batch.numel()
            if isinstance(uncertainty, dict) and "exit_layer" in uncertainty:
                el = uncertainty["exit_layer"]
                if torch.is_tensor(el):
                    exit_layers.extend(el.detach().cpu().tolist())
                elif isinstance(el, (list, tuple, np.ndarray)):
                    exit_layers.extend(list(el))
                else:
                    exit_layers.extend([int(el)] * src_batch.size(0))

    n_samples = len(src)
    avg_exit = (sum(exit_layers) / len(exit_layers)) if len(exit_layers) > 0 else None

    layer_hist = {}
    if len(exit_layers) > 0:
        vals, counts = np.unique(exit_layers, return_counts=True)
        total = float(len(exit_layers))
        for v, c in zip(vals, counts):
            layer_hist[int(v)] = float(c) / total

    return (n_samples, n_tok, avg_exit, layer_hist)


@wall_time
def timed_osr_stage(args, test_dataset, open_dataset):

    (uncertainties_test,
     pred_labels_test,
     gt_labels_test,
     exit_layers_test) = test_1(
        args, test_dataset, device=args.device, tag="test"
    )

    if open_dataset:
        (uncertainties_open,
         pred_labels_open,
         gt_labels_open,
         exit_layers_open) = test_1(
            args, open_dataset, device=args.device, tag="open"
        )
    else:
        uncertainties_open = np.array([])
        pred_labels_open = np.array([])
        gt_labels_open = np.array([])
        exit_layers_open = np.array([])

    return (uncertainties_test, pred_labels_test, gt_labels_test, exit_layers_test,
            uncertainties_open, pred_labels_open, gt_labels_open, exit_layers_open)


def plot_uncertainty_distribution(test_unc, open_unc,
                                  threshold=None,
                                  save_path="uncertainty_distribution_new3.png"):

    plt.figure(figsize=(10, 6))

    plt.hist(test_unc, bins=50, density=True, alpha=0.6,
             color='blue', label='Known (Test)')

    if len(open_unc) > 0:
        plt.hist(open_unc, bins=50, density=True, alpha=0.6,
                 color='red', label='Unknown (Open)')

    if threshold is not None:
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                    label=f'Threshold: {threshold:.4f}')

    max_test_unc = float(np.max(test_unc))
    plt.axvline(x=max_test_unc, color='black', linestyle='--', linewidth=2,
                label=f'Max Test Unc: {max_test_unc:.4f}')

    plt.xlabel("Uncertainty Score (Mutual Information)")
    plt.ylabel("Density")
    plt.title("Uncertainty Distribution: Known vs Unknown")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path)
    plt.close()




def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"],
                        default="first")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"],
                        default="bert")
    parser.add_argument("--soft_targets", action='store_true')
    parser.add_argument("--soft_alpha", type=float, default=0.5)

    parser.add_argument("--is_open", action='store_true')
    parser.add_argument("--open_path", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.8)


    parser.add_argument("--exit_layers", type=str, default="",
                        help="1-based layers, e.g. '4,6,8,10,12'")
    parser.add_argument("--enable_early_exit", action="store_true")
    parser.add_argument("--early_exit_metric", type=str, default="mutual_info",
                        choices=["mutual_info", "entropy_of_expected",
                                 "expected_entropy", "k_over_alpha0"])
    parser.add_argument("--early_exit_threshold", type=float, default=0.05)
    parser.add_argument("--train_exits", action="store_true")
    parser.add_argument("--min_exit_layer", type=int, default=1,
                        )
    parser.add_argument("--normalize_collected", action="store_true",
                        )
    parser.add_argument("--no-normalize_collected", dest="normalize_collected",
                        action="store_false")

    parser.add_argument('--only_test', action='store_true',
                        help="Only run evaluation, skip training.")


    parser.add_argument("--exit_loss_weight", type=float, default=0.3,
                        help="Weight of early-exit losses when train_exits is enabled.")


    parser.add_argument("--early_exit_low", type=float, default=None,)

    parser.add_argument("--early_exit_high", type=float, default=None,)



    parser.add_argument("--osr_mode", action="store_true",)


    args = parser.parse_args()
    args = load_hyperparam(args)


    args.exit_layers = _parse_exit_layers(args.exit_layers)
    if args.min_exit_layer < 1:
        args.min_exit_layer = 1

    if args.early_exit_low is None:
        args.early_exit_low = args.early_exit_threshold

    set_seed(args.seed)


    args.labels_num = count_labels_num(args.train_path) + 1
    UNKNOWN_LABEL = args.labels_num - 1


    args.tokenizer = str2tokenizer[args.tokenizer](args)


    model = Classifier(args)
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)


    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([ex[0] for ex in trainset])
    tgt = torch.LongTensor([ex[1] for ex in trainset])
    seg = torch.LongTensor([ex[2] for ex in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([ex[3] for ex in trainset])
    else:
        soft_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model


    if not args.only_test:
        print("Start training.")
        total_loss, best_result = 0.0, 0.0
        for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
            model.train()
            for i, (src_b, tgt_b, seg_b, soft_b) in enumerate(
                    batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                loss = train_model(args, model, optimizer, scheduler,
                                   src_b, tgt_b, seg_b, soft_b)
                total_loss += loss.item()
                if (i + 1) % args.report_steps == 0:
                    print(f"Epoch id: {epoch}, Training steps: {i+1}, "
                          f"Avg loss: {total_loss/args.report_steps:.3f}")
                    total_loss = 0.0

            result = evaluate(args, read_dataset(args, args.dev_path))
            if result[0] > best_result:
                best_result = result[0]
                save_model(model, args.output_model_path)


    if args.test_path is not None:
        print("Test set evaluation.")
        state = safe_torch_load(args.output_model_path, map_location="cpu")
        if torch.cuda.device_count() > 1:
            missing, unexpected = model.module.load_state_dict(state, strict=False)
        else:
            missing, unexpected = model.load_state_dict(state, strict=False)
        print("[CKPT][best] missing:", missing, "unexpected:", unexpected)


    print("Test set evaluation (train-style, no EE).")
    evaluate(args, read_dataset(args, args.test_path), True)

    test_dataset = read_dataset(args, args.test_path)
    open_dataset = read_dataset(args, args.open_path) if args.open_path else []

    print("\n===== Closed-set evaluation on Test set (with Early Exit) =====")

    model_inst = args.model.module if isinstance(args.model, torch.nn.DataParallel) else args.model
    ee_restore_closed = bool(getattr(model_inst, "enable_early_exit", False))


    setattr(model_inst, "osr_mode", False)

    model_inst.enable_early_exit = True
    print("\n[Closed-set][With Early Exit]")
    (n_ee, n_tok_ee, avg_exit_ee, layer_hist_ee), dt_test_ee = timed_infer(args, test_dataset)
    print(f"Inference time on TEST (EE): {dt_test_ee:.3f}s")
    acc_test_ee, confusion_ee = evaluate_infer(args, test_dataset, print_confusion_matrix=True)
    print(f"[Closed-set][EE]    Acc = {acc_test_ee:.4f}")

    if avg_exit_ee is not None:
        L = getattr(args, 'layers_num', 12)
        print(f"Avg exit layer (TEST): {avg_exit_ee:.2f} / {L} "
              f"(≈ {L / max(1e-6, avg_exit_ee):.2f}x)")
        print("Per-layer exit ratio on TEST set (EE enabled):")
        for lid in range(1, L + 1):
            ratio = layer_hist_ee.get(lid, 0.0)
            print(f"  Layer {lid:2d}: {ratio * 100:6.2f}%")
    else:
        print("Warning: no exit_layer information collected on TEST; "
              "cannot compute per-layer exit ratio.")

    if open_dataset:
        print("\n[Open-set][With Early Exit] (only for exit-layer stats)")
        (n_open_ee, n_tok_open_ee, avg_exit_open_ee, layer_hist_open_ee), dt_open_ee = timed_infer(args, open_dataset)
        print(f"Inference time on OPEN (EE): {dt_open_ee:.3f}s")
        L = getattr(args, 'layers_num', 12)
        if avg_exit_open_ee is not None:
            print(f"Avg exit layer (OPEN): {avg_exit_open_ee:.2f} / {L} "
                  f"(≈ {L / max(1e-6, avg_exit_open_ee):.2f}x)")
            print("Per-layer exit ratio on OPEN set (EE enabled):")
            for lid in range(1, L + 1):
                ratio = layer_hist_open_ee.get(lid, 0.0)
                print(f"  Layer {lid:2d}: {ratio * 100:6.2f}%")
        else:
            print("Warning: no exit_layer information collected on OPEN; "
                  "cannot compute per-layer exit ratio.")

    model_inst.enable_early_exit = ee_restore_closed

    print('Test:', len(test_dataset))
    if open_dataset:
        print('Open:', len(open_dataset))


    per_layer_high_path = "per_layer_high_thr.json"
    if os.path.exists(per_layer_high_path):
        with open(per_layer_high_path, "r") as f:
            args.per_layer_high = json.load(f)
        print("Loaded per-layer high thresholds:", args.per_layer_high)
    else:
        args.per_layer_high = None

    per_layer_low_path = "per_layer_low_thr.json"
    if os.path.exists(per_layer_low_path):
        with open(per_layer_low_path, "r") as f:
            args.per_layer_low = json.load(f)
        print("Loaded per-layer low thresholds:", args.per_layer_low)
    else:
        args.per_layer_low = None
        print("No per_layer_low_thr.json found,  early_exit_low.")


    setattr(model_inst, "osr_mode", True)

    (timed_outputs, dt_osr) = timed_osr_stage(args, test_dataset, open_dataset)
    (uncertainties_test, pred_labels_test, gt_labels_test, exit_layers_test,
     uncertainties_open, pred_labels_open, gt_labels_open, exit_layers_open) = timed_outputs

    print("\n===== System-level cost (OSR stage) =====")
    print(f"Forward time for Test+Open (OSR / test_1): {dt_osr:.3f}s")

    if open_dataset and len(uncertainties_open) > 0:
        uncertainties_flat = np.concatenate([
            np.array(uncertainties_test).flatten(),
            np.array(uncertainties_open).flatten()
        ])
        pred_labels_flat = np.concatenate([
            np.array(pred_labels_test).flatten(),
            np.array(pred_labels_open).flatten()
        ])
        gt_labels_flat = np.concatenate([
            np.array(gt_labels_test).flatten(),
            np.array(gt_labels_open).flatten()
        ])
        exit_layers_flat = np.concatenate([
            np.array(exit_layers_test).flatten(),
            np.array(exit_layers_open).flatten()
        ])
    else:
        uncertainties_flat = np.array(uncertainties_test).flatten()
        pred_labels_flat = np.array(pred_labels_test).flatten()
        gt_labels_flat = np.array(gt_labels_test).flatten()
        exit_layers_flat = np.array(exit_layers_test).flatten()

    print("Shapes:", uncertainties_flat.shape, pred_labels_flat.shape, gt_labels_flat.shape)

    L_all = getattr(args, 'layers_num', 12)
    if exit_layers_test.size > 0:
        print("\nPer-exit-layer ratio (OSR stage, TEST):")
        for lid in range(1, L_all + 1):
            ratio = np.mean(exit_layers_test == lid)
            print(f"  Layer {lid:2d}: {ratio * 100:6.2f}%")

    if open_dataset and exit_layers_open.size > 0:
        print("\nPer-exit-layer ratio (OSR stage, OPEN):")
        for lid in range(1, L_all + 1):
            ratio = np.mean(exit_layers_open == lid)
            print(f"  Layer {lid:2d}: {ratio * 100:6.2f}%")

    if exit_layers_flat.size > 0:
        print("\nPer-exit-layer ratio (OSR stage, TEST+OPEN):")
        for lid in range(1, L_all + 1):
            ratio = np.mean(exit_layers_flat == lid)
            print(f"  Layer {lid:2d}: {ratio * 100:6.2f}%")

    if open_dataset and len(uncertainties_open) > 0:

        uncertainties_test_sorted = np.sort(uncertainties_test)
        uncertainties_open_sorted = np.sort(uncertainties_open)
        all_uncertainties = np.sort(np.unique(
            np.concatenate([uncertainties_test_sorted, uncertainties_open_sorted])
        ))
        cdf_test_interp = np.searchsorted(
            uncertainties_test_sorted, all_uncertainties, side='right'
        ) / len(uncertainties_test_sorted)
        cdf_open_interp = np.searchsorted(
            uncertainties_open_sorted, all_uncertainties, side='right'
        ) / len(uncertainties_open_sorted)


        max_fpr = 0.005
        min_tpr = 0.30

        best_high = None
        for u, cdf_c, cdf_o in zip(all_uncertainties, cdf_test_interp, cdf_open_interp):
            fpr = 1.0 - cdf_c
            tpr = 1.0 - cdf_o
            if fpr <= max_fpr and tpr >= min_tpr:
                best_high = u
                break

        if best_high is None:

            best_high = float(np.quantile(uncertainties_open_sorted, 0.9))

        print(f"[Heuristic] Suggested early_exit_high = {best_high:.4f} "
              f"(FPR_closed <= {max_fpr}, TPR_open >= {min_tpr})")

        with open("suggested_high_thr.txt", "w") as f:
            f.write(f"{best_high:.6f}\n")


        cdf_diff = np.abs(cdf_test_interp - cdf_open_interp)
        max_index = np.argmax(cdf_diff)
        best_threshold = all_uncertainties[max_index]
        max_gap = cdf_diff[max_index]
        print(f"Best threshold = {best_threshold:.4f}, Max gap = {max_gap:.4f}")

        plt.figure()
        plt.plot(all_uncertainties, cdf_test_interp, label='Test CDF')
        plt.plot(all_uncertainties, cdf_open_interp, label='Open CDF')
        plt.axvline(best_threshold, linestyle='--',
                    label=f'Max Gap @ {best_threshold:.4f}')
        plt.scatter([best_threshold], [cdf_test_interp[max_index]], zorder=5)
        plt.scatter([best_threshold], [cdf_open_interp[max_index]], zorder=5)
        plt.xlabel('Uncertainty')
        plt.ylabel('Cumulative Probability')
        plt.title('Test vs Open Uncertainty CDF')
        plt.legend()
        plt.grid(True)
        plt.savefig('Max_CDF_Gap.png')

        print("\n[Plotting] Generating Uncertainty Distribution Histogram...")
        plot_uncertainty_distribution(
            test_unc=uncertainties_test,
            open_unc=uncertainties_open,
            threshold=best_threshold,
            save_path="uncertainty_distribution.png"
        )
    else:
        best_threshold = args.threshold


    for target_layer in [4, 8, 12]:

        mask_test_l = (exit_layers_test == target_layer)
        test_unc_l = np.array(uncertainties_test)[mask_test_l]


        mask_open_l = (exit_layers_open == target_layer)
        open_unc_l = np.array(uncertainties_open)[mask_open_l]

        print(f"[Layer {target_layer}] TEST : {mask_test_l.sum()}, "
              f"OPEN : {mask_open_l.sum()}")

        if mask_test_l.sum() > 0 or mask_open_l.sum() > 0:
            plot_uncertainty_distribution(
                test_unc=test_unc_l,
                open_unc=open_unc_l,
                threshold=best_threshold,
                save_path=f"uncertainty_distribution_layer{target_layer}.png"
            )

    if open_dataset and len(uncertainties_open) > 0:
        n_test = len(uncertainties_test)
        n_open = len(uncertainties_open)
        binary_labels = np.array([0] * n_test + [1] * n_open)
        if len(np.unique(binary_labels)) < 2:
            auc_val = float('nan')
        else:
            auc_val = roc_auc_score(binary_labels, uncertainties_flat)
            print('AUC:', auc_val)

        metrics = evaluate_open_set_metrics(
            uncertainties=uncertainties_flat,
            pred_labels=pred_labels_flat,
            gt_labels=gt_labels_flat,
            threshold=best_threshold,
            known_label_num=args.labels_num
        )
        print("\n===== Open-set & Closed-set metrics (with rejection, overall) =====")
        for key, val in metrics.items():
            print(f"{key}: {val:.4f}")

        print("\n===== Per-exit-layer OSR metrics (with rejection, Test+Open) =====")
        unique_layers = sorted(set(exit_layers_flat.tolist()))
        UNKNOWN_CLASS = args.labels_num - 1

        for lid in unique_layers:
            mask_l = (exit_layers_flat == lid)
            if not np.any(mask_l):
                continue

            u_l = uncertainties_flat[mask_l]
            pred_l = pred_labels_flat[mask_l]
            gt_l = gt_labels_flat[mask_l]

            metrics_l = evaluate_open_set_metrics(
                uncertainties=u_l,
                pred_labels=pred_l,
                gt_labels=gt_l,
                threshold=best_threshold,
                known_label_num=args.labels_num
            )

            pred_with_reject_l = np.where(u_l > best_threshold, UNKNOWN_CLASS, pred_l)
            gt_l_np = np.array(gt_l)

            is_unknown_l = (gt_l_np == UNKNOWN_CLASS)
            is_closed_l = ~is_unknown_l

            closed_total_l = int(is_closed_l.sum())
            open_total_l = int(is_unknown_l.sum())

            closed_correct_l = int(
                np.sum(pred_with_reject_l[is_closed_l] == gt_l_np[is_closed_l])
            ) if closed_total_l > 0 else 0

            open_correct_l = int(
                np.sum(pred_with_reject_l[is_unknown_l] == UNKNOWN_CLASS)
            ) if open_total_l > 0 else 0

            ratio_l = np.mean(mask_l)

            print(f"  Layer {int(lid):2d}: ratio={ratio_l * 100:6.2f}%")
            print(
                f"    Closed-Acc={metrics_l['Closed-set Acc']:.4f}"
                f"(Correct/Total={closed_correct_l}/{closed_total_l}), "
                f"Closed-F1={metrics_l['Closed-set F1']:.4f}, "
                f"Closed-P={metrics_l['Closed-set Precision']:.4f}, "
                f"Closed-R={metrics_l['Closed-set Recall']:.4f}"
            )
            print(
                f"    Open-Acc={metrics_l['Open-set Acc']:.4f}"
                f"(Correct/Total={open_correct_l}/{open_total_l}), "
                f"Open-F1={metrics_l['Open-set F1']:.4f}, "
                f"Open-P={metrics_l['Open-set Precision']:.4f}, "
                f"Open-R={metrics_l['Open-set Recall']:.4f}"
            )

        fpr, tpr, thresholds = roc_curve(binary_labels, uncertainties_flat)
        roc_auc_val = calc_auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('ROC_curve.png')
        plt.show()

    total_system_time = dt_test_ee + dt_osr
    print("\n===== System-level cost  =====")
    print(f"[Part 1] Closed-set EE forward on TEST: {dt_test_ee:.3f}s")
    print(f"[Part 2] OSR stage forward (Test+Open, test_1): {dt_osr:.3f}s")
    print(f"--> Total System Forward Time: {total_system_time:.3f}s")


if __name__ == "__main__":
    main()
