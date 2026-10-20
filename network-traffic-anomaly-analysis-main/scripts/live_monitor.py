import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from scapy.all import sniff, IP, TCP, UDP

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "dataset/processed"
BUFFER_FILE   = PROCESSED_DIR / "live_buffer.json"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

with open(PROCESSED_DIR / 'windowed_columns.json') as f:
    EXPECTED_COLS = json.load(f)

try:
    df_n = pd.read_csv(PROCESSED_DIR / 'normal_windowed.csv')
    NORMAL_MEANS = df_n.mean(numeric_only=True).to_dict()
    df_t = pd.read_csv(PROCESSED_DIR / 'test_windowed_results.csv')
    ATTACK_MEANS = df_t[df_t['window_attack']==1].mean(numeric_only=True).to_dict()
except Exception as e:
    print(f"Warning: {e}")
    NORMAL_MEANS = {}
    ATTACK_MEANS = {}

packet_count = 0
start_time   = time.time()
window_id    = int(time.time())
ip_ttls      = []
pkt_sizes    = []
protos       = set()
services     = defaultdict(int)
states       = defaultdict(int)

def process_packet(packet):
    global packet_count, start_time, window_id
    global ip_ttls, pkt_sizes, protos, services, states

    packet_count += 1

    if IP in packet:
        protos.add(packet[IP].proto)
        ip_ttls.append(packet[IP].ttl)
        pkt_sizes.append(len(packet))

        if TCP in packet or UDP in packet:
            port = packet[TCP].dport if TCP in packet else packet[UDP].dport
            if port == 53 or (TCP in packet and packet[TCP].sport == 53):
                services['dns'] += 1
            elif port in [80, 443]:
                services['http'] += 1
            elif port in [20, 21]:
                services['ftp'] += 1
            elif port == 22:
                services['ssh'] += 1
            else:
                services['-'] += 1

            if UDP in packet or (TCP in packet and packet[TCP].flags == 'S'):
                states['INT'] += 1
            else:
                states['CON'] += 1

    duration = time.time() - start_time
    if packet_count >= 100 or (duration > 5.0 and packet_count > 5):
        flush_window(duration)

def flush_window(duration):
    global packet_count, start_time, window_id
    global ip_ttls, pkt_sizes, protos, services, states

    if packet_count == 0:
        return

    rate      = packet_count / max(duration, 0.001)
    ttl_mean  = sum(ip_ttls)   / len(ip_ttls)   if ip_ttls   else 0
    size_mean = sum(pkt_sizes) / len(pkt_sizes) if pkt_sizes else 0
    int_ratio = states['INT']  / max(packet_count, 1)

    base = ATTACK_MEANS if rate > 500 else NORMAL_MEANS
    row  = {col: base.get(col, 0.0) for col in EXPECTED_COLS}

    row['window_id']        = window_id
    row['rate_mean']        = np.log1p(rate)
    row['sttl_mean']        = 254 if rate > 200 else ttl_mean
    row['sbytes_mean']      = np.log1p(size_mean)
    row['dbytes_mean']      = np.log1p(0 if int_ratio > 0.8 else size_mean * 0.8)
    row['proto_nunique']    = len(protos)
    row['dur_mean']         = np.log1p(duration)
    row['service_dns_mean'] = services['dns']  / packet_count
    row['service_http_mean']= services['http'] / packet_count
    row['service_-_mean']   = services['-']    / packet_count
    row['state_INT_mean']   = states['INT']    / packet_count
    row['state_CON_mean']   = states['CON']    / packet_count
    row['window_attack']    = 0
    row['raw_rate']         = rate

    buffer = []
    if BUFFER_FILE.exists():
        try:
            with open(BUFFER_FILE, 'r') as f:
                buffer = json.load(f)
        except json.JSONDecodeError:
            pass

    buffer.append(row)
    buffer = buffer[-20:]

    with open(BUFFER_FILE, 'w') as f:
        json.dump(buffer, f)

    print(f"[{time.strftime('%H:%M:%S')}] Window {window_id} "
          f"({packet_count} pkts, {rate:.1f} pkts/s)")

    packet_count = 0
    start_time   = time.time()
    window_id   += 1
    ip_ttls.clear()
    pkt_sizes.clear()
    protos.clear()
    services.clear()
    states.clear()

if __name__ == "__main__":
    print("Starting live network monitor...")
    print(f"Writing to {BUFFER_FILE}")
    print("Press Ctrl+C to stop.")

    with open(BUFFER_FILE, 'w') as f:
        json.dump([], f)

    try:
        sniff(prn=process_packet, store=False)
    except PermissionError:
        print("Run as Administrator to capture packets.")
    except RuntimeError as e:
        if "winpcap" in str(e).lower():
            print("Install Npcap from https://npcap.com/#download")
        else:
            print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Stopped.")