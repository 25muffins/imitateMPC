from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# PREDECLARED PARAMETERS -- FREEZE BEFORE LOOKING AT RESULTS
# ============================================================

# Physical output ranges used to make vx and vy comparable.
CONTROL_SCALE = np.array([1.0, 1.0, 4.0], dtype=np.float64)

# Useful-effect threshold.
USEFUL_MULTIPLIER = 3.0
PHYSICAL_MINIMUM = 0.1

# Bootstrap settings
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 12345


# ============================================================
# HELPERS
# ============================================================

def vector(df, prefix):
    return df[
        [f"{prefix}_vx", f"{prefix}_vy", f"{prefix}_w"]
    ].to_numpy(dtype=np.float64)


def normalized_norm(delta):
    """
    Current M1 definition:
    normalized Euclidean norm using vx and vy only.
    """
    return np.linalg.norm(
        delta[:, :3] / CONTROL_SCALE[None, :],
        axis=1
    )


def bootstrap_median_ci(x, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """
    Percentile bootstrap 95% CI for the median.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        x,
        size=(n_boot, len(x)),
        replace=True
    )

    medians = np.median(samples, axis=1)

    return (
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5))
    )


def summarize_effect(effect, useful):
    """
    Calculate the core M1 descriptive statistics.
    """
    effect = np.asarray(effect, dtype=np.float64)
    useful = np.asarray(useful, dtype=bool)

    median = np.median(effect)
    q1 = np.percentile(effect, 25)
    q3 = np.percentile(effect, 75)
    mean = np.mean(effect)
    p90 = np.percentile(effect, 90)
    p95 = np.percentile(effect, 95)

    ci_low, ci_high = bootstrap_median_ci(effect)

    return {
        "n": len(effect),
        "mean": mean,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "p90": p90,
        "p95": p95,
        "useful_count": int(useful.sum()),
        "useful_fraction": float(useful.mean()),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("fixed_anchor_audit.csv")

base = vector(df, "base")

useful_threshold = PHYSICAL_MINIMUM

print("predeclared useful threshold:", useful_threshold)
print("control scale:", CONTROL_SCALE)


# ============================================================
# M1 AUDIT
# ============================================================

all_results = []


for intervention in ["left", "right", "nonext"]:

    changed = vector(df, intervention)

    # Difference between factual and counterfactual teacher command
    delta = changed - base

    # Magnitude of teacher response
    effect = normalized_norm(delta)

    # Useful intervention
    useful = effect > useful_threshold

    print("\n" + "=" * 60)
    print(intervention)
    print("=" * 60)

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    summary = summarize_effect(effect, useful)

    print("anchors:", summary["n"])

    print("effect mean:",
          summary["mean"])

    print("effect median:",
          summary["median"])

    print("effect IQR:",
          summary["iqr"])

    print("effect Q1:",
          summary["q1"])

    print("effect Q3:",
          summary["q3"])

    print("effect p90:",
          summary["p90"])

    print("effect p95:",
          summary["p95"])

    print(
        "median 95% CI:",
        f"[{summary['ci95_low']}, "
        f"{summary['ci95_high']}]"
    )

    print(
        "useful:",
        f"{summary['useful_count']}/{summary['n']}"
    )

    print(
        "useful fraction:",
        summary["useful_fraction"]
    )


    # --------------------------------------------------------
    # Phase-specific results
    # --------------------------------------------------------

    for phase_name in ["pre_switch", "post_switch"]:

        m = df["phase"].to_numpy() == phase_name

        if not m.any():
            continue

        phase_effect = effect[m]
        phase_useful = useful[m]

        phase_summary = summarize_effect(
            phase_effect,
            phase_useful
        )

        print("\n", phase_name)

        print(
            "n =",
            phase_summary["n"]
        )

        print(
            "mean =",
            phase_summary["mean"]
        )

        print(
            "median =",
            phase_summary["median"]
        )

        print(
            "IQR =",
            phase_summary["iqr"]
        )

        print(
            "p90 =",
            phase_summary["p90"]
        )

        print(
            "useful =",
            f"{phase_summary['useful_count']}/"
            f"{phase_summary['n']}"
        )

        print(
            "useful fraction =",
            phase_summary["useful_fraction"]
        )

        print(
            "median 95% CI =",
            f"[{phase_summary['ci95_low']}, "
            f"{phase_summary['ci95_high']}]"
        )

        all_results.append({
            "intervention": intervention,
            "phase": phase_name,
            **phase_summary
        })


# ============================================================
# SAVE SUMMARY
# ============================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv("Counterfactual_data.csv", index=False)

print("\n\nFINAL M1 SUMMARY")
print(results_df.to_string(index=False))