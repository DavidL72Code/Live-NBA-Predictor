# Analysis Fix Five — Temperature Scaling

## Experiment

Tested a single global temperature parameter on the current 20-feature model:

- 3,443 training games
- 492 calibration games
- 984 untouched test games
- Exact active-run parameters: 100 trees, max depth 5, random state 42, 8 threads
- Exact active-run game split: 3,443 train / 492 calibration / 984 test games
- Temperature fitted by minimizing calibration-set log loss

Temperature scaling transforms each raw probability in logit space:

```text
P_calibrated = sigmoid(logit(P_raw) / T)
```

The fitted temperature was **T = 0.982**. Since T < 1, it very slightly sharpens
predictions away from 50% without changing their ordering.

## Results

| Method | Brier score | Log loss | ROC-AUC |
|---|---:|---:|---:|
| Raw XGBoost | **0.164379** | **0.485918** | **0.833245** |
| Isotonic | 0.165037 | 0.487925 | 0.832644 |
| Temperature scaling | 0.164496 | 0.486220 | **0.833245** |

Temperature scaling worsened Brier by **0.000117** and log loss by **0.000302**
versus raw output. Isotonic was worse by **0.000658** Brier. Temperature
scaling preserved ranking exactly, but that does not offset the regression.

## Decision

Do not switch production serving. On the exact active test split, raw XGBoost
is the best result. Temperature scaling remains implemented as an experiment,
but should only be reconsidered after a larger time-based evaluation shows a
repeatable improvement.
