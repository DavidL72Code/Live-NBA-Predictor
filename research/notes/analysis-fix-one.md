# Analysis Fix One — Isotonic Regression Calibration

## The Fix We Tried

The XGBoost win-probability model's reliability diagram showed systematic
miscalibration at the extremes: the 0–10% bucket had predictions averaging
2.7% but actual home-win rates of 10.5% (gap −0.077), and the 80–90% bucket
underpredicted by −0.044. The 20–80% middle range was already well-calibrated
(|gap| ≤ 0.031).

The proposed fix was to layer an **isotonic regression calibrator** on top of
the raw XGBoost output. We ran two variants:

| Variant | Calibration data | Approach |
|---|---|---|
| Single-fold | 123 games (10% split) | Train 80%, calibrate 10%, test 10% |
| Out-of-fold (OOF) | 1,107 games (5-fold CV) | Hold out 10% test, generate OOF predictions on 90%, fit isotonic on all 90% |

---

## Why We Thought It Was a Good Idea

**Why isotonic over Platt scaling:**
Platt scaling fits a single sigmoid `P_cal = 1 / (1 + exp(a·logit(p) + b))`. To
correct the 0–20% underconfidence it must shift the sigmoid left, which
simultaneously pushes the already-good 30–60% predictions into overconfidence.
Fixing one end breaks the other — you rob Peter to pay Paul.

Isotonic regression is piecewise: it fits an independent non-decreasing step
function over the probability range. Each bucket is corrected only as much as
the data in that bucket requires. Buckets that are already well-calibrated get
barely touched; the 0–20% region gets a meaningful upward correction. That was
the theory.

**Why OOF over a single calibration fold:**
A single 10% holdout gives isotonic only 123 games. With such a small sample,
the fitted step function captures noise from which 123 games happened to land
in that fold. The OOF approach generates predictions for every training game
through cross-validation (~1,107 games) without sacrificing model quality,
since the final XGBoost is retrained on all 90%.

---

## Resulting Outcome

| Metric | Raw XGBoost | Single-fold isotonic | OOF isotonic |
|---|---|---|---|
| **Brier score** | 0.1689 | 0.1705 | 0.1690 |
| **vs raw** | — | **−1.0% (worse)** | **−0.1% (neutral)** |
| ROC-AUC | 0.8295 | 0.8285 | 0.8294 |

**Reliability gaps by bucket (Predicted − Actual):**

| Bucket | Raw gap | Single-fold cal gap | OOF cal gap |
|---|---|---|---|
| 0–10% | −0.077 | −0.081 | **−0.072** ← slight improvement |
| 10–20% | −0.037 | +0.039 ← flipped sign | −0.015 |
| 40–50% | −0.018 | **+0.093** ← collapsed | −0.015 |
| 50–60% | +0.031 | **+0.068** ← collapsed | +0.032 |
| 70–80% | −0.005 | +0.011 | −0.019 |
| 90–100% | −0.016 | +0.028 | −0.036 |

Single-fold isotonic destroyed the 40–70% range (gaps jumped from ~0.02 to
0.07–0.09). The OOF approach eliminated that collapse entirely. The 0–10%
bucket improved modestly with OOF (−0.077 → −0.072). Neither approach
improved the overall Brier score.

---

## Why This Happened

**The fundamental misspecification:**
Isotonic regression assumes every data point is an independent observation.
In NBA play-by-play data, every row within a game shares the same binary
outcome label — `home_win` is a game-level result applied to all ~487 rows.

Within a single game:
- Predictions range from ~5% (first possession, score 0–0) up to ~95% (final
  minutes, home team up by 15).
- All those rows carry the same label: 1 if the home team won, 0 if not.

Isotonic regression sees this as: "predictions in range [0.05, 0.50] appearing
in a game labeled 1" and maps that range toward 1. The step function
effectively learns which probability ranges co-occur with which game outcomes
in the calibration fold. That is a property of the specific 123 (or 1,107)
games in the calibration set, not a universal feature of the model's output.

When the test set has a different mix of game outcomes at those probability
ranges (which it will), the step function misfires.

**Why single-fold was catastrophic:**
With only 123 calibration games, the step function was extremely sensitive to
which games happened to land there. A run of home-team wins in the 40–60%
prediction range mapped those probabilities to ~1.0; the test set's different
game mix then received badly wrong predictions.

**Why OOF was much better but still neutral:**
With 1,107 games the step function was far more stable — no systematic
collapse. But the core misspecification (within-game temporal structure) still
prevented it from producing a genuine improvement. Improvements in early-game
buckets (0–20%) were offset by small degradations in late-game buckets
(70–100%), where the final model's raw predictions were already accurate and
isotonic introduced noise.

---

## Next Steps

**Option A — Segment-aware calibration (recommended first attempt):**
Fit the calibrator only on rows in the final 12 minutes of regulation (Q4,
≤720 seconds remaining). At that stage the outcome is largely determined,
the temporal structure is less confounding, and calibration accuracy matters
most for a live product.

```
calibration set = dev rows where seconds_remaining <= 720
```

Fit isotonic on those rows only; apply to all rows at serve time.
This is a one-function change to `train_oof` — add a mask before
`calibrator.fit(oof_preds, oof_labels)`.

**Option B — Temperature scaling:**
A single scalar T: `P_cal = sigmoid(logit(P_raw) / T)`. Only one degree of
freedom; essentially impossible to overfit even on 123 games. Fits T to
minimize NLL on the calibration set. Less expressive than isotonic (won't fix
bucket-specific errors) but provably can't make things worse the way isotonic
can when overfit. Good sanity check before committing to segment-aware work.

**Option C — Ship the raw output:**
The raw model scores Brier 0.1689, AUC 0.8295. The 20–80% range — where most
live-game queries land — is already calibrated to within ±0.04. The 0–10%
underconfidence (predictions averaging 2.5% vs 10.5% actual win rate) is
driven by early first-quarter rows where the game is effectively still a
coin-flip; a user-facing display could simply suppress win-probability output
until the first score or until a few minutes have elapsed.

**Not worth pursuing:**
- More data (another season of backfill) — the root cause is structural, not
  sample size. OOF with 1,107 games already proved this.
- Platt scaling — as established, it will fix one end and break the other given
  the gap profile of this model.
- Global isotonic on all rows without segment filtering — we've now run this
  experiment twice; the answer is stable.
