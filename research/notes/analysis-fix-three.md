# Analysis Fix Three — Game-Level Aggregation for Isotonic Calibration

## The Fix We Tried

Fixes One and Two both failed because isotonic regression treated each row as
an independent observation, but every row within a game shares the same binary
label. The proposed fix was to **aggregate OOF predictions to the game level**
before fitting isotonic — one mean prediction per game, one label per game —
so the calibrator sees 1,107 genuinely independent data points instead of
538,905 rows with correlated labels.

```python
game_agg = dev_df.groupby("game_id").agg(
    _mean_pred=("oof_pred", "mean"),
    _label=("home_win", "first"),   # all rows share the same label
)
calibrator.fit(game_agg["_mean_pred"], game_agg["_label"])
```

The calibrator was then applied to individual row predictions at serve time as
before: `y_prob_cal = calibrator.predict(y_prob_raw)`.

---

## Why We Thought It Was a Good Idea

The root cause identified in Fixes One and Two was that isotonic regression's
independence assumption is violated: 538,905 rows give it the illusion of a
large calibration dataset, but there are really only 1,107 independent outcomes.
By aggregating to one (mean prediction, label) pair per game, we respect the
actual unit of independence. The step function would be fitted on data whose
labels are genuinely one-per-observation, eliminating the within-game
correlation entirely.

---

## Resulting Outcome

| Metric | Raw XGBoost | OOF global (Fix 1) | Q4-only (Fix 2) | Game-level (Fix 3) |
|---|---|---|---|---|
| **Brier score** | 0.1687 | 0.1690 | 0.1693 | **0.1897** |
| **vs raw** | — | −0.1% | −0.3% | **−12.45%** |
| ROC-AUC | 0.8298 | 0.8294 | 0.8283 | 0.8250 |

**Reliability gaps — game-level calibrated:**

| Bucket | Raw rows | Raw gap | Cal rows | Cal gap |
|---|---|---|---|---|
| 0–10% | 7,056 | −0.080 | **18,002** | **−0.164** |
| 10–20% | 4,337 | −0.029 | 4,006 | −0.231 |
| 20–30% | 4,770 | −0.006 | **10** | −0.252 |
| 30–40% | 4,819 | +0.017 | 3,149 | −0.151 |
| 40–50% | 5,291 | −0.020 | 1,441 | −0.000 |
| 50–60% | 7,741 | +0.041 | 6,030 | +0.066 |
| 60–70% | 9,023 | +0.017 | 3,647 | +0.086 |
| 70–80% | 5,284 | −0.004 | **3** | **−0.277** |
| 80–90% | 3,848 | −0.035 | 8,693 | +0.166 |
| 90–100% | 7,631 | −0.016 | 3,674 | +0.211 |

The calibrated distribution is completely distorted: 20–30% has 10 rows (from
4,770), 70–80% has 3 rows (from 5,284), and the 0–10% bucket swelled from
7,056 to 18,002.

---

## Why This Happened

**Domain mismatch between training and serving.**
Game-level mean predictions cluster in a narrow range — roughly [0.38, 0.65].
Even in blowouts, the mean prediction across the full game arc is moderate
because the first half often starts near 0.5 before predictions diverge. A
game the home team wins by 20 might have a mean prediction of 0.62; a game they
lose by 20 might have a mean of 0.40.

The isotonic step function is therefore fitted only over [0.38, 0.65]. But at
serve time it is applied to individual row predictions that span [0.02, 0.98].

**`out_of_bounds="clip"` pins out-of-range predictions to the boundary outputs.**
Scikit-learn's `IsotonicRegression(out_of_bounds="clip")` clips the *input* X
to [min_train, max_train] before looking up the output. So:

- Any row prediction < 0.38 → clipped to 0.38 → output = isotonic(0.38)
- Any row prediction > 0.65 → clipped to 0.65 → output = isotonic(0.65)

If isotonic(0.38) ≈ 0.05 (games with that mean prediction lose ~80% of the
time), then every one of the thousands of early-game rows with predictions below
0.38 gets mapped to 0.05 — all collapse into the 0–10% bucket. Symmetrically,
all high-confidence rows pile into the 80–90% bucket. The middle of the
distribution (20–70%) gets nearly emptied because those raw predictions map
directly through the step function, but few game means land in the extreme
regions to give the step function meaningful anchors there.

**The conceptual fix is right but the execution is wrong.**
Treating games as atomic units IS the correct framing. The failure is in
applying a game-average calibrator to row-level predictions — those live in
different spaces. To be valid, the calibration and serving would need to happen
at the same level of aggregation: compute one probability per game (e.g. at
halftime), calibrate that, and serve that. That is a product architecture
decision, not a model tweak.

---

## What We Have Learned Across All Three Fixes

| Fix | Approach | Brier vs raw | Failure mode |
|---|---|---|---|
| 1 (single-fold) | Row-level isotonic, 123 cal games | −1.0% | Overfits to fold's game mix |
| 1 (OOF) | Row-level isotonic, 1,107 cal games | −0.1% | Within-game label correlation |
| 2 (Q4-only) | Row-level isotonic, Q4 rows only | −0.3% | Covariate shift Q4→all quarters |
| 3 (game-level) | Game-mean isotonic, 1,107 games | −12.5% | Domain mismatch: mean space vs row space |

Every isotonic variant has made the Brier worse. The failures are structurally
distinct, which confirms the raw model's calibration is the ceiling for
post-hoc correction without a fundamental change in approach.

---

## Next Steps

**Option A — Temperature scaling (strongly recommended):**
A single scalar T: `P_cal = sigmoid(logit(P_raw) / T)`. Fits one parameter
against the NLL on OOF predictions. Has none of the failure modes above:
- No domain mismatch: applies to row predictions the same way it was fitted.
- No bucket collapse: a sigmoid transformation cannot redistibute rows into
  artificial piles.
- No overfitting: one degree of freedom cannot memorise game-mix noise.
- Interpretable: T > 1 compresses predictions toward 0.5 (less confident);
  T < 1 sharpens them.

This is the controlled baseline that should have been tried before isotonic.

**Option B — Ship the raw output:**
Brier 0.1687, AUC 0.8298. Three rounds of post-hoc calibration have not
improved on the raw model. The 20–80% range (where most live-game queries
land) has |gap| ≤ 0.041. The 0–10% underconfidence is a Q1 artefact that can
be mitigated at the display layer. The model is production-ready as-is.

**Not worth pursuing:**
- Any further isotonic variant — the structural mismatches between row-level
  predictions and game-level labels have now been demonstrated conclusively
  from four angles.
- Game-level serving + game-level calibration — would require rearchitecting
  the product to surface one probability per game rather than a live ticker.
