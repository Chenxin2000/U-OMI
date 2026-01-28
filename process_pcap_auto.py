
import argparse
import os
import random
from time import time

import dpkt
import socket
from multiprocessing import Pool
from typing import List, Tuple


BYTE_LEN = 64
FLOW_LEN = 8


class get_flows():
    def __init__(self, read_path=None, output_file=None, filter_empty=False,
                 bi_direction=True, multi_processing=False, process_num=os.cpu_count(),
                 store_label_dict=True):
        self.saved_flows = []
        self.read_path = read_path
        self.output_file = output_file
        self.filter_empty = filter_empty
        self.bi_direction = bi_direction
        self.app_label_dict = {}
        self.app_ips = None
        self.multi_processing = multi_processing
        self.process_num = process_num
        self.store_label_dict = store_label_dict
        self.app_dir_list = []
        self.app_pcap_list = []

        if self.read_path is not None:
            self.data_set_name = self.read_path.split('/')[-1]
        else:
            self.data_set_name = None

        self.apps_need = None

    def set_read_path(self, read_path):
        self.read_path = read_path
        self.data_set_name = self.read_path.split('/')[-1]

    def add_ips_filter(self, file_name):
        with open(file_name, 'r') as f:
            self.app_ips = f.read()
            self.app_ips = eval(self.app_ips)

    def add_apps_filter(self, file_name):
        with open(file_name, 'r') as f:
            self.apps_need = f.read()
            self.apps_need = eval(self.apps_need)

    def merge_hex_pairs(self, hex_list):
        merged = []
        for i in range(0, len(hex_list) - 1, 1):
            merged.append(hex_list[i] + hex_list[i + 1])
        return merged

    def output_payload(self, ip):
        tcpudp = ip.data
        payload = tcpudp.data
        ip_bytes = ["%02x" % int(p) for p in ip.pack().replace(ip.data.pack(), b"")[:-8]]
        tcp_bytes = ["%02x" % int(p) for p in ip.data.pack()[:]]
        full_bytes = ip_bytes + tcp_bytes

        ip_length = len(full_bytes)

        if ip_length > BYTE_LEN:
            full_bytes = full_bytes[:BYTE_LEN]

        merged_bytes = self.merge_hex_pairs(full_bytes)
        formatted = ' '.join(merged_bytes)
        return formatted

    def analyze_packet(self, ts, pkt, flow_dict, app_type, link_type):
        if link_type == 101:
            ip = dpkt.ip.IP(pkt)
        else:
            eth = dpkt.ethernet.Ethernet(pkt)
            if not isinstance(eth.data, dpkt.ip.IP):
                return
            ip = eth.data

        if not isinstance(ip.data, dpkt.tcp.TCP):
            return

        if len(ip.data.data) == 0 and self.filter_empty:
            return

        src_ip = socket.inet_ntoa(ip.src)
        dst_ip = socket.inet_ntoa(ip.dst)
        src_port = ip.data.sport
        dst_port = ip.data.dport

        if self.app_ips is not None:
            if src_ip not in self.app_ips[app_type] and dst_port not in self.app_ips[app_type]:
                return

        key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"

        if key in flow_dict:
            if len(flow_dict[key][0]) < FLOW_LEN:
                flow_dict[key][0].append(str(ip.len))
                packet_byte = self.output_payload(ip)
                flow_dict[key][1].append(packet_byte)
        else:
            key_rev = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}"
            if key_rev in flow_dict and self.bi_direction:
                if len(flow_dict[key_rev][0]) < FLOW_LEN:
                    flow_dict[key_rev][0].append(str(-ip.len))
                    packet_byte = self.output_payload(ip)
                    flow_dict[key_rev][1].append(packet_byte)
            else:
                packet_byte = self.output_payload(ip)
                flow_dict[key] = ([str(ip.len)], [packet_byte])

    def save_flows(self, saved_flows, flow_dict, label):
        for flows in flow_dict.values():
            byte_stream = str(" ".join(flows[1]))
            str_item = '\t'.join([str(label), byte_stream])
            saved_flows.append(str_item)
        print(f'[Info] Current app: {self.app_label_dict[label]}, '
              f'label: {label}, total flows: {len(saved_flows)}')

    def output_flows(self, file_name, saved_flows):
        direction = 'bi_direction' if self.bi_direction else 'single_direction'
        print('[Info] Saving captured flows...')
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w+') as f:
            f.write('\n'.join(saved_flows))
        print(f'[Info] Successfully save {len(saved_flows)} {direction} flows!')

    def processing_pcap(self, file_name, flow_dict, app_type=None):
        file = open(file_name, 'rb')
        magic_head = file.read(4)
        file.seek(0, 0)
        if magic_head == b'\n\r\r\n':
            packets = dpkt.pcapng.Reader(file)
        elif magic_head == b'\xd4\xc3\xb2\xa1' or magic_head == b'\xa1\xb2\xc3\xd4':
            packets = dpkt.pcap.Reader(file)
        else:
            print("[DEBUG in PcapUtils] It is not a pcap or pcapng file")
            file.close()
            return

        link_type = packets.datalink()
        for ts, pkt in packets:
            try:
                self.analyze_packet(ts, pkt, flow_dict, app_type, link_type)
            except Exception as e:
                print(e)
                continue
        file.close()

    def processing_app(self, app_type, app_pcap_list, label, output=False):
        flow_dict = {}
        saved_flows = []
        for pcap_file in app_pcap_list:
            pcap_file_path = os.path.join(self.read_path, self.app_dir_list[label], pcap_file)
            try:
                self.processing_pcap(pcap_file_path, flow_dict, app_type)
            except Exception as e:
                print(e)
                continue
        self.save_flows(saved_flows, flow_dict, label)
        if output:
            out_path = os.path.join(self.output_file, f"{app_type}.txt")
            self.output_flows(out_path, saved_flows)
        return saved_flows

    def processing_apps(self):
        all_flows = []


        self.app_dir_list = os.listdir(self.read_path)

        if self.apps_need is not None:

            for i in self.app_dir_list:
                if i not in self.apps_need:
                    self.app_dir_list.remove(i)

        for label, app_type in enumerate(self.app_dir_list):
            self.app_pcap_list.append(os.listdir(os.path.join(self.read_path, app_type)))
            self.app_label_dict[label] = app_type

        if self.store_label_dict:
            direction = 'bi_direction' if self.bi_direction else 'single_direction'
            with open(f"{self.data_set_name}_{direction}_label_dict.txt", 'w+') as f:
                f.write(str(self.app_label_dict))

        if self.multi_processing:
            with Pool(processes=self.process_num) as pool:
                result_objects = [
                    pool.apply_async(
                        self.processing_app,
                        args=(self.app_label_dict[label], self.app_pcap_list[label], label, True)
                    )
                    for label in range(len(self.app_dir_list))
                ]
                for result_object in result_objects:
                    all_flows += result_object.get()
        else:
            for label in range(len(self.app_dir_list)):
                all_flows += self.processing_app(self.app_label_dict[label],
                                                 self.app_pcap_list[label],
                                                 label,
                                                 False)





def write_to_tsv(lines: List[str], filename: str) -> None:

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        f.write("label\ttext_a\n")
        for line in lines:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                label, text_a = parts
            else:
                label, text_a = parts[0], ""
            f.write(f"{label}\t{text_a}\n")
    print(f"  {len(lines)} to {filename}")


def split_file_lines(lines: List[str]) -> Tuple[List[str], List[str], List[str]]:

    if not lines:
        return [], [], []

    total = len(lines)
    valid_size = max(1, int(total * 0.1))
    test_size = max(1, int(total * 0.1))
    train_size = total - valid_size - test_size

    random.shuffle(lines)
    train = lines[:train_size]
    valid = lines[train_size:train_size + valid_size]
    test = lines[train_size + valid_size:]

    return train, valid, test


def process_all_files(directory: str,
                      min_lines: int = 0,
                      max_lines: int = None) -> Tuple[List[str], List[str], List[str]]:

    all_train = []
    all_valid = []
    all_test = []

    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    print(f"[Step 2]  {len(txt_files)} ")
    count_num = 0

    for filename in txt_files:
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            '''if len(lines) < min_lines:
                continue
            if len(lines) > max_lines:
                lines = lines[:max_lines]
            '''
            count_num += 1
            train, valid, test = split_file_lines(lines)

            all_train.extend(train)
            all_valid.extend(valid)
            all_test.extend(test)

            print(f" {filename}: {len(lines)}  → "
                  f"({len(train)}), ({len(valid)}), ({len(test)})")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    print(f" {count_num}  #####################\n")
    return all_train, all_valid, all_test





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PCAP -> flows(txt) -> BERT TSV"
    )


    parser.add_argument('-read_path', type=str, default='/data/CICIOT2023',)
    parser.add_argument('-output_file', type=str,
                        default='/home/chenchuanxin/datasets/CICIOT2023/CICIOT202364_8_out',)

    parser.add_argument('-ips_filter', type=str,
                        default='/home/chenchuanxin/datasets/CICIOT2023/Unique_sip.json')
    parser.add_argument('-apps_filter', type=str,
                        default='/home/chenchuanxin/datasets/CICIOT2023/CICIOT202364_8.txt')
    parser.add_argument('-use_ips_filter', action='store_true',)
    parser.add_argument('-use_apps_filter', action='store_true',)

    parser.add_argument('-filter_empty', type=bool, default=True)
    parser.add_argument('-bi_direction', type=bool, default=True)
    parser.add_argument('-multi_processing', type=bool, default=True)
    parser.add_argument('-process_num', type=int, default=os.cpu_count() // 2)
    parser.add_argument('-store_label_dict', type=bool, default=True)


    parser.add_argument('-final_out_dir', type=str,
                        default='/home/chenchuanxin/BCBA/datasets/CICIOT2023',)
    parser.add_argument('-tsv_tag', type=str, default='CICIOT202364_8',)
    parser.add_argument('-min_lines', type=int, default=0,)
    parser.add_argument('-max_lines', type=int, default=None,)

    args = parser.parse_args()


    random.seed(42)


    os.makedirs(args.output_file, exist_ok=True)
    print(f"[Info] Step1 : {args.output_file}")

    flows_getter = get_flows(read_path=args.read_path,
                             output_file=args.output_file,
                             filter_empty=args.filter_empty,
                             bi_direction=args.bi_direction,
                             multi_processing=args.multi_processing,
                             process_num=args.process_num,
                             store_label_dict=args.store_label_dict)

    if args.use_ips_filter and args.ips_filter:
        flows_getter.add_ips_filter(file_name=args.ips_filter)
    if args.use_apps_filter and args.apps_filter:
        flows_getter.add_apps_filter(file_name=args.apps_filter)

    t0 = time()
    flows_getter.processing_apps()
    print(f"[Step 1]  {time() - t0:.1f} s， txt : {args.output_file}")


    print(f"\n[Step 2] {args.output_file}  txt ...")
    train, valid, test = process_all_files(args.output_file,
                                           min_lines=args.min_lines,
                                           max_lines=args.max_lines)

    if not train and not valid and not test:
        print("no Step 2 end")
        exit(0)


    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    total = len(train) + len(valid) + len(test)
    print("\n final")
    print(f"{len(train)} ({len(train) / total:.2%})")
    print(f" {len(valid)} ({len(valid) / total:.2%})")
    print(f"{len(test)}  ({len(test) / total:.2%})")

    train_path = os.path.join(args.final_out_dir, f"train_{args.tsv_tag}.tsv")
    valid_path = os.path.join(args.final_out_dir, f"valid_{args.tsv_tag}.tsv")
    test_path = os.path.join(args.final_out_dir, f"test_{args.tsv_tag}.tsv")

    write_to_tsv(train, train_path)
    write_to_tsv(valid, valid_path)
    write_to_tsv(test, test_path)

    print("\n final！")
    print(f"{train_path}")
    print(f"{valid_path}")
    print(f"{test_path}")
