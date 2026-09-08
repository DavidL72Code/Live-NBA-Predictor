# Analysis Fix Two — Segment-Aware Isotonic Calibration (Q4 Only)

## The Fix We Tried

Fix One established that global isotonic regression fails because every row
within a game shares one binary label while predictions span the full 5%→95%
range across the game arc. The proposed solution was to restrict the calibration
fit to **Q4 rows only** (`seconds_remaining ≤ 720`, last 12 minutes), where the
game outcome is largely decided and predictions are already high-confidence. The
calibrator would then be applied to all rows at serve time.

Implementation: one mask added to `train_oof` before `calibrator.fit()`:

```python
cal_mask = dev_df["seconds_remaining"].values <= 720
calibrator.fit(oof_preds[cal_mask], oof_labels[cal_mask])
```

Calibration rows used: **142,333** (Q4 rows across 1,107 games, via OOF).

---

## Why We Thought It Was a Good Idea

In Q4 the temporal confound that broke global isotonic weakens considerably:

- A prediction of 5% in Q4 almost always means the home team is down by a
  large margin with little time left — the label (0) is nearly certain.
- A prediction of 95% in Q4 means the home team is comfortably ahead — the
  label (1) is nearly certain.
- The step function should learn a cleaner mapping because prediction values and
  outcome labels are tightly correlated within Q4, unlike Q1 where a 5%
  prediction might still belong to a game that ultimately ends 1.

The 0–10% underconfidence problem (raw gap −0.080) was expected to improve most,
since late-game rows with predictions below 10% genuinely correspond to losing
teams.

---

## Resulting Outcome

| Metric | Raw XGBoost | OOF global (Fix 1) | Q4-only OOF (Fix 2) |
|---|---|---|---|
| **Brier score** | 0.1687 | 0.1690 | 0.1693 |
| **vs raw** | — | −0.1% (neutral) | **−0.3% (worse)** |
| ROC-AUC | 0.8298 | 0.8294 | 0.8283 |

**Reliability gaps by bucket (Predicted − Actual):**

| Bucket | Raw rows | Raw gap | Q4-cal rows | Q4-cal gap | Change |
|---|---|---|---|---|---|
| 0–10% | 7,056 | −0.080 | 6,353 | −0.071 | ✓ improved |
| 10–20% | 4,337 | −0.029 | 5,789 | −0.026 | ✓ improved |
| 20–30% | 4,770 | −0.006 | 3,708 | −0.012 | ✗ slightly worse |
| 30–40% | 4,819 | +0.017 | 7,071 | −0.009 | ✓ improved |
| 40–50% | 5,291 | −0.020 | **834** | **−0.065** | ✗ collapsed |
| 50–60% | 7,741 | +0.041 | 8,914 | +0.037 | ✓ marginal |
| 60–70% | 9,023 | +0.017 | 7,918 | +0.024 | ✗ slightly worse |
| 70–80% | 5,284 | −0.004 | 8,522 | +0.007 | ✗ small flip |
| 80–90% | 3,848 | −0.035 | 3,584 | −0.053 | ✗ worse |
| 90–100% | 7,631 | −0.016 | 4,243 | −0.029 | ✗ worse |

The most significant failure: the 40–50% bucket shrank from 5,291 rows to just
834, and its gap worsened from −0.020 to −0.065. The calibrator is pushing large
numbers of predictions out of the 40–50% range — a sign of "stretching" at the
midpoint.

---

## Why This Happened

**Q4 predictions skew toward the extremes.**
In Q4, close games are the minority. Most rows have predictions either below
30% (home team losing by enough to be unlikely to recover) or above 70% (home
team ahead). The isotonic step function fitted on Q4 OOF data therefore has
most of its calibration signal at the tails — there are relatively few Q4 rows
in the 40–50% range to anchor that part of the curve.

**The step function stretches the middle.**
When applied to test rows from Q1–Q3 — which have many more moderate
predictions (40–60%) reflecting genuine early-game uncertainty — the Q4-derived
step function maps those moderate values outward toward the tails. This collapses
the 40–50% bucket from 5,291 rows to 834 and creates the large −0.065 gap.

**Training segment ≠ serving segment.**
Fitting on Q4 and serving on all quarters is a covariate shift problem. The
joint distribution (prediction, label) in Q4 is fundamentally different from
Q1–Q3: Q4 has higher-confidence predictions and more extreme label agreement.
The calibrator learned from one distribution but was asked to correct another.

**Why the tails still didn't fully fix:**
Even in Q4, the 0–10% and 80–90% gaps remained (−0.071 and −0.053
respectively). These segments still involve temporal context (a team 20 points
down in Q3 entering Q4 vs Q4 with 30 seconds left are both "Q4" but very
different), so the step function still captures some game-mix noise.

---

## What We Have Learned Across Both Fixes

After two calibration experiments (three variants total), a consistent picture
has emerged:

| Approach | Cal rows | Brier vs raw | Key failure |
|---|---|---|---|
| Single-fold isotonic | ~60k (123 games) | −1.0% | Step fn overfits to fold's game mix |
| OOF global isotonic | ~539k (1,107 games) | −0.1% | Within-game label correlation |
| OOF Q4-only isotonic | ~142k (1,107 games, Q4) | −0.3% | Covariate shift Q4→all quarters |

Isotonic regression is structurally mismatched to this data. The label
(`home_win`) is a game-level outcome; every prediction emitted by the model
during a game is a per-row estimate of that same outcome. Isotonic regression
cannot distinguish "this prediction is low because it's a Q1 toss-up" from
"this prediction is low because it's Q4 and the home team is losing by 20."
No segmentation scheme fully resolves this without also restricting serving to
the same segment.

---

## Next Steps

**Option A — Temperature scaling (recommended next):**
A single scalar T: `P_cal = sigmoid(logit(P_raw) / T)`. One parameter, fits
in seconds, essentially impossible to overfit regardless of sample size or game
structure. It compresses or expands the entire probability range uniformly —
it won't fix bucket-specific errors, but it cannot create the stretching or
collapse artefacts that isotonic produces. Fitting T to minimise NLL on the
OOF predictions would directly address the 50–60% overconfidence (+0.041 gap)
if T > 1 (slight compression). This is the proper controlled baseline before
any further complexity.

**Option B — Separate serving by segment:**
If Q4 calibration is applied **only at serve time for Q4 rows** (and raw output
is used for Q1–Q3), the covariate shift disappears. The 40–50% collapse would
not occur because Q4 rows rarely land there. The trade-off: the calibrator
provides no correction for the majority of game time. This is only worthwhile
if the product specifically surfaces win probability in late-game contexts.

**Option C — Accept the raw output:**
Brier 0.1687, AUC 0.8298. Two rounds of calibration experiments have confirmed
that no standard post-hoc calibrator improves on the raw model given the
within-game label structure. The 20–80% range is well-calibrated (|gap| ≤
0.041). The 0–10% underconfidence is an early-game artefact that can be
addressed at the display layer (suppress or caveat win probability in the first
few minutes of Q1).
