"""Generate a synthetic Network Traffic Attribution dataset.

Models the distinction between traffic originating from the real attacker
source (label = 1) and traffic from spoofed / decoy sources (label = 0).

Output: data/sample/attribution_sample.csv with 14 attribution features
plus a binary `label` column.

Synthesis rules (informed by Born & Gustafson 2010, Mirkovic & Reiher 2004,
and the CIC-DDoS2019 dataset structure):

  Real attacker (label = 1):
    - Consistent OS-specific TTL (centered on 64 or 128, small jitter).
    - Low TTL variance.
    - Low source-IP entropy (one or a few stable IPs).
    - Stable TCP window size.
    - Consistent TCP options hash.
    - Bounded packet size distribution.
    - Predictable inter-arrival timing.
    - Low retransmission ratio.

  Spoofed / decoy (label = 0):
    - Randomized TTL spanning many values (decoy traffic mixes OSes).
    - High TTL variance.
    - High source-IP entropy (many random IPs).
    - Variable / unstable TCP window sizes.
    - Multiple options hashes per flow.
    - Wide packet size distribution.
    - Jittery inter-arrival timing.
    - Elevated retransmission ratio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _real_flow(rng: np.random.Generator) -> dict[str, float]:
    base_ttl = rng.choice([64.0, 128.0, 255.0])
    return {
        "ttl_mean":             base_ttl + rng.normal(0, 0.7),
        "ttl_variance":         abs(rng.normal(0.5, 0.4)),
        "src_ip_entropy":       max(0.0, rng.normal(1.6, 0.4)),         # narrow set of IPs
        "tcp_window_size_mean": rng.normal(64240, 1500),
        "tcp_window_size_std":  abs(rng.normal(120, 80)),
        "tcp_options_hash":     float(rng.integers(1_000_000, 9_999_999)),  # one signature per source
        "packet_size_mean":     rng.normal(820, 60),
        "packet_size_std":      abs(rng.normal(40, 15)),
        "interarrival_mean":    abs(rng.normal(0.012, 0.003)),
        "interarrival_variance":abs(rng.normal(0.0008, 0.0004)),
        "flow_duration":        abs(rng.normal(32, 6)),
        "packets_per_second":   abs(rng.normal(85, 12)),
        "bytes_per_packet":     rng.normal(820, 55),
        "retransmission_ratio": max(0.0, rng.normal(0.012, 0.008)),
    }


def _spoof_flow(rng: np.random.Generator) -> dict[str, float]:
    return {
        "ttl_mean":             rng.uniform(15, 240),                  # any TTL
        "ttl_variance":         abs(rng.normal(35, 12)),                # huge jitter
        "src_ip_entropy":       min(7.5, abs(rng.normal(5.5, 0.6))),    # many random IPs
        "tcp_window_size_mean": rng.uniform(2000, 65000),
        "tcp_window_size_std":  abs(rng.normal(3500, 1200)),
        "tcp_options_hash":     float(rng.integers(1_000_000, 9_999_999)),
        "packet_size_mean":     rng.uniform(80, 1500),
        "packet_size_std":      abs(rng.normal(380, 130)),
        "interarrival_mean":    abs(rng.normal(0.045, 0.025)),
        "interarrival_variance":abs(rng.normal(0.012, 0.006)),
        "flow_duration":        abs(rng.normal(14, 11)),
        "packets_per_second":   abs(rng.normal(220, 90)),               # bursty decoy floods
        "bytes_per_packet":     rng.uniform(60, 1500),
        "retransmission_ratio": max(0.0, rng.normal(0.085, 0.04)),
    }


def _mixed_real_flow(rng: np.random.Generator) -> dict[str, float]:
    """A 'hard' real-attacker flow that mimics some spoof characteristics —
    e.g. a real attacker behind a forwarding proxy, or partially randomized
    TCP window. Adds class overlap so the dataset is not trivially separable.
    """
    d = _real_flow(rng)
    # Inject noise on two or three features that an attacker would also see.
    d["src_ip_entropy"]      = abs(rng.normal(3.0, 0.6))
    d["tcp_window_size_std"] = abs(rng.normal(900, 300))
    d["interarrival_variance"] = abs(rng.normal(0.004, 0.002))
    return d


def _mixed_spoof_flow(rng: np.random.Generator) -> dict[str, float]:
    """A spoof flow that has been carefully shaped to mimic a real source."""
    d = _spoof_flow(rng)
    d["ttl_mean"]      = rng.choice([64.0, 128.0]) + rng.normal(0, 3)
    d["ttl_variance"]  = abs(rng.normal(3, 1.5))
    d["src_ip_entropy"] = abs(rng.normal(2.4, 0.5))
    return d


def build_dataset(n_real: int, n_spoof: int, seed: int = 42,
                  hard_fraction: float = 0.08) -> pd.DataFrame:
    """Build a balanced attribution dataset.

    `hard_fraction` controls how many flows are drawn from the 'mixed'
    distributions that overlap with the opposite class — this creates
    realistic class confusion instead of producing trivially separable data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    n_hard_real  = int(n_real  * hard_fraction)
    n_hard_spoof = int(n_spoof * hard_fraction)
    for _ in range(n_real - n_hard_real):
        d = _real_flow(rng); d["label"] = 1
        rows.append(d)
    for _ in range(n_hard_real):
        d = _mixed_real_flow(rng); d["label"] = 1
        rows.append(d)
    for _ in range(n_spoof - n_hard_spoof):
        d = _spoof_flow(rng); d["label"] = 0
        rows.append(d)
    for _ in range(n_hard_spoof):
        d = _mixed_spoof_flow(rng); d["label"] = 0
        rows.append(d)
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-real", type=int, default=5000)
    parser.add_argument("--n-spoof", type=int, default=5000)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent
                    / "data" / "sample" / "attribution_sample.csv"),
    )
    args = parser.parse_args()

    df = build_dataset(args.n_real, args.n_spoof)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print(df.sample(5, random_state=1))


if __name__ == "__main__":
    main()
