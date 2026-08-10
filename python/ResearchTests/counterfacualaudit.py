import numpy as np

# ── factual (test1) and counterfactual (test2) outputs ───────────
# columns: vx, vy, omega (global, unnormalized)
test1 = np.array([
    [3.583777386905554,  10.773101911388338, -1.666946644890057],
    [2.300534346164941,   6.927725603577850, -0.782158379049563],
    [1.477397147438843,   4.467893672653091, -0.366665036196614],
    [0.949739946568365,   2.901641148524049, -0.171170714200180],
    [0.612029709552547,   1.915717604700241, -0.078378698082019],
    [0.396726595362888,   1.312934549340739, -0.032613567305813],
    [0.260768109346811,   0.972742507141259, -0.006416314891817],
    [0.176965085633435,   0.827096202583602,  0.015930512390472],
    [0.025710635125543,   0.169373188928574,  0.095671704023112],
])

test2 = np.array([
    [3.590895155746041,  10.748775329643664, -1.666593905859655],
    [2.309083232934911,   6.898530508515665, -0.781598277908168],
    [1.489066492716140,   4.427999133060865, -0.365558805038869],
    [0.966868479364176,   2.843064492281798, -0.168854427294815],
    [0.638047426592322,   1.826743102646708, -0.073461109202984],
    [0.436838379442150,   1.175770486219159, -0.022144253390108],
    [0.322994762849718,   0.759952953986220,  0.015887987491844],
    [0.273756399392594,   0.496128223855111,  0.063450640712228],
])

# ── normalize to common output space ─────────────────────────────
# match Python training normalization
target_div = np.array([30, 30, np.pi])
test1_norm = test1 / target_div
test2_norm = test2 / target_div

# ── compute d_i = ||u_counterfactual - u_factual||_2 ─────────────
# test1 and test2 have different row counts — use minimum
n   = min(len(test1_norm), len(test2_norm))
d_i = np.linalg.norm(test2_norm[:n] - test1_norm[:n], axis=1)

print("Per-anchor effect d_i (normalized L2):")
for i, d in enumerate(d_i):
    print(f"  anchor {i+1}: {d:.5f}")

# ── useful threshold ──────────────────────────────────────────────
# repeat floor — approximate solver noise
# use smallest d_i as proxy for noise floor
repeat_floor_p95 = np.percentile(d_i, 5)
delta_deadband   = 0.001
delta_min        = 0.005
delta_useful     = max(5 * repeat_floor_p95, delta_deadband, delta_min)

print(f"\nRepeat floor p95:  {repeat_floor_p95:.5f}")
print(f"delta_useful:      {delta_useful:.5f}")

# ── count useful and directional ──────────────────────────────────
useful = d_i > delta_useful
k      = useful.sum()

print(f"\nEffects d_i:")
print(f"  Median:     {np.median(d_i):.5f}")
print(f"  IQR:        [{np.percentile(d_i,25):.5f}, "
      f"{np.percentile(d_i,75):.5f}]")
print(f"  95th pct:   {np.percentile(d_i,95):.5f}")
print(f"  Min:        {d_i.min():.5f}")
print(f"  Max:        {d_i.max():.5f}")
print(f"\nUseful k/N: {k}/{n} ({100*k/n:.1f}%)")

# ── per component analysis ────────────────────────────────────────
diff = test2_norm[:n] - test1_norm[:n]
print(f"\nPer-component mean absolute difference:")
print(f"  vx:    {np.abs(diff[:,0]).mean():.5f}")
print(f"  vy:    {np.abs(diff[:,1]).mean():.5f}")
print(f"  omega: {np.abs(diff[:,2]).mean():.5f}")

# ── interpretation ────────────────────────────────────────────────
print(f"\nInterpretation:")
print(f"  {k}/{n} anchors show useful response to g2 change")
if k/n > 0.5:
    print("  TEACHER LOOKS AHEAD — MPC uses g2 before switch")
    print("  → anticipatory architecture justified")
    print("  → history + attention should help student")
elif k/n < 0.2:
    print("  TEACHER IS REACTIVE — MPC ignores g2 before switch")
    print("  → explains why baseline beats history models")
    print("  → student only needs current state + g1")
else:
    print("  WEAK ANTICIPATION — MPC partially uses g2")
    print("  → architecture choice matters less")