"""Feature documentation for the Network Traffic Attribution pipeline.

The pipeline consumes 14 numeric flow-level features directly. Each feature
captures a signal that distinguishes a real attacker source from a spoofed or
decoy source. They are produced upstream of this code (by a packet-capture
analyzer or by `generate_sample.py`) and persisted as columns of the CSV that
`preprocess.load_dataset` reads.
"""

from __future__ import annotations

# Names and one-line intuitions for the 14 attribution features.
ATTRIBUTION_FEATURES = {
    "ttl_mean":                "Average TTL across packets in the flow. Real sources show consistent OS-specific TTLs (64, 128, 255).",
    "ttl_variance":            "TTL variance — spoofed traffic often shows large jumps; real sources are stable.",
    "src_ip_entropy":          "Shannon entropy of source-IP bytes. Decoy floods come from many randomized addresses → high entropy.",
    "tcp_window_size_mean":    "Mean advertised TCP window. Specific to the sender's OS — fingerprint signal.",
    "tcp_window_size_std":     "Stddev of the TCP window. Real sources hold near-constant; spoofed varies wildly.",
    "tcp_options_hash":        "Integer hash of the TCP options string — coarse OS fingerprint.",
    "packet_size_mean":        "Mean packet size — payloads from a real attacker tend to cluster.",
    "packet_size_std":         "Stddev of packet sizes — high for chaotic spoofed mixes.",
    "interarrival_mean":       "Mean inter-arrival time between packets.",
    "interarrival_variance":   "Variance of inter-arrival times — real sources have predictable cadence.",
    "flow_duration":           "Total flow duration in seconds.",
    "packets_per_second":      "Packets/sec for the flow.",
    "bytes_per_packet":        "Mean bytes per packet.",
    "retransmission_ratio":    "Fraction of packets that were retransmissions — anomalously high for spoofs.",
}


def feature_names() -> list[str]:
    return list(ATTRIBUTION_FEATURES.keys())


if __name__ == "__main__":
    for name, desc in ATTRIBUTION_FEATURES.items():
        print(f"  {name:<26} {desc}")
