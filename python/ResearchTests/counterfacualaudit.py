from __future__ import annotations

import numpy as np
import pandas as pd


# Physical output ranges used only to make vx, vy, and omega comparable.
# Freeze these values before opening the audit results.
CONTROL_SCALE = np.array([1, 1], dtype=np.float64)
USEFUL_MULTIPLIER = 3.0
PHYSICAL_MINIMUM = .1  # normalized control norm; predeclare this value


def vector(df, prefix):
    return df[[f"{prefix}_vx", f"{prefix}_vy", f"{prefix}_w"]].to_numpy()


def normalized_norm(delta):
    return np.linalg.norm(delta[:, :2] / CONTROL_SCALE[None, :], axis=1)


df = pd.read_csv("fixed_anchor_audit.csv")

base = vector(df, "base")

useful_threshold = PHYSICAL_MINIMUM

print("predeclared useful threshold:", useful_threshold)


for intervention in ["left", "right", "nonext"]:
    changed = vector(df, intervention)
    effect = normalized_norm(changed - base)
    useful = effect > useful_threshold

    print("\n", intervention)
    print("anchors:", len(effect))
    print("effect mean:", np.mean(effect))
    print("effect p90:", np.percentile(effect, 90))
    print("useful fraction:", useful.mean())

    # Phase-specific results answer the real handoff question.
    for phase_name in ["pre_switch", "post_switch"]:
        m = df["phase"].to_numpy() == phase_name
        if m.any():
            print(
                phase_name,
                "n=", int(m.sum()),
                "mean=", float(np.mean(effect[m])),
                "useful_fraction=", float(np.mean(useful[m])),
            )