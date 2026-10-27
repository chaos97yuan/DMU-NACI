## Table X. Cross-system point forecasting (2025 cross-year, median of 5 seeds)

| System | Capacity | Configuration | RMSE (kW) | nRMSE | MAE (kW) | nMAE |
|---|---:|---|---:|---:|---:|---:|
| Site 18 | 5.2 kW | fixed-tilt mono-Si | 0.286 | 0.0551 | 0.110 | 0.0211 |
| Site 1B | 23.4 kW | dual-axis mono-Si tracker | 0.867 | 0.0370 | 0.406 | 0.0174 |

## Table Y. Cross-system probabilistic forecasting (2025, median of 5 seeds)

| System | Method | PICP | PINAW | IS (kW) | NIS | RCE1D | RCE3D |
|---|---|---:|---:|---:|---:|---:|---:|
| Site 18 | A0 | 0.944 | 0.187 | 1.567 | 0.301 | 0.0791 | 0.0615 |
| Site 18 | A1 | 0.901 | 0.099 | 0.874 | 0.168 | 0.0135 | 0.0074 |
| Site 18 | A2 | 0.895 | 0.075 | 0.664 | 0.128 | 0.0469 | 0.0373 |
| Site 18 | A3 | 0.900 | 0.075 | 0.654 | 0.126 | 0.0369 | 0.0226 |
| Site 18 | CQR | 0.943 | 0.092 | 0.753 | 0.145 | 0.0599 | 0.0599 |
| Site 1B | A0 | 0.950 | 0.160 | 4.983 | 0.213 | 0.0711 | 0.0564 |
| Site 1B | A1 | 0.918 | 0.093 | 3.097 | 0.132 | 0.0224 | 0.0193 |
| Site 1B | A2 | 0.855 | 0.060 | 2.574 | 0.110 | 0.0647 | 0.0561 |
| Site 1B | A3 | 0.898 | 0.070 | 2.530 | 0.108 | 0.0232 | 0.0128 |
| Site 1B | CQR | 0.903 | 0.067 | 2.683 | 0.115 | 0.0417 | 0.0296 |

## Residual pool audit (Site 1B)

| Component | n | MAE (kW) | RMSE | abs-e q50 | q90 | q95 |
|---|---:|---:|---:|---:|---:|---:|
| E22_replay | 487,075 | 0.450 | 0.978 | 0.113 | 1.179 | 2.095 |
| E23_replay | 454,715 | 0.351 | 0.817 | 0.100 | 0.784 | 1.638 |
| E24_OOF | 120,490 | 0.394 | 0.993 | - | - | - |

E22 vs E23 (2023 low-power operational state retained): 2023 MAE 0.351 kW < 2022 0.450 kW -- the reduced-power state yielded SMALLER absolute replay residuals (lower attainable power), recorded as-is.
Predicted sigma (seed 42): 2022 median 0.139 kW / q90 1.841; 2023 median 0.127 / q90 1.312

## Frozen gamma / initial q (Site 1B)

- seed 42: gamma_A3 0.05, gamma_A1 0.05, q_abs 1.8754 kW, q_norm 1.4579
- seed 123: gamma_A3 0.005, gamma_A1 0.05, q_abs 1.9056 kW, q_norm 1.4475
- seed 2026: gamma_A3 0.002, gamma_A1 0.05, q_abs 1.8723 kW, q_norm 1.4642
- seed 3407: gamma_A3 0.01, gamma_A1 0.05, q_abs 1.9832 kW, q_norm 1.3746
- seed 8888: gamma_A3 0.02, gamma_A1 0.05, q_abs 1.8678 kW, q_norm 1.4156

## Bootstrap (Site 1B 2025, day-block paired, 2000 resamples)

- A3-vs-A0 IS: diff -2.4462 [-2.5294, -2.3645]  SIGNIFICANT (5/5 seeds)
- A3-vs-A0 NIS: diff -0.1045 [-0.1081, -0.1010]  SIGNIFICANT (5/5 seeds)
- A3-vs-A0 RCE1D: diff -0.0479 [-0.0506, -0.0449]  SIGNIFICANT (5/5 seeds)
- A3-vs-A2 IS: diff -0.0204 [-0.0555, +0.0107]  tied (2/5 seeds)
- A3-vs-A2 NIS: diff -0.0009 [-0.0024, +0.0005]  tied (2/5 seeds)
- A3-vs-A2 RCE1D: diff -0.0427 [-0.0476, -0.0384]  SIGNIFICANT (5/5 seeds)
- A3-vs-CQR IS: diff -0.0113 [-0.0558, +0.0324]  tied (3/5 seeds)
- A3-vs-CQR NIS: diff -0.0005 [-0.0024, +0.0014]  tied (3/5 seeds)
- A3-vs-CQR RCE1D: diff -0.0176 [-0.0202, -0.0149]  SIGNIFICANT (5/5 seeds)

## Scientific verdict

1. Coverage: A3 PICP 0.898 vs nominal 0.90 -> retained.
2. Normalization efficiency: A3 NIS 0.108 vs A0 0.213 (5/5 seeds significant) -> reproduced.
3. ACI stability: A3 RCE1D 0.0232 vs A0 0.0711, A2 0.0647, CQR 0.0417 (5/5 significant) -> reproduced.
4. vs CQR: IS diff CI includes 0 (-0.0558..+0.0324) -> globally statistically tied; A3 better on rolling stability (5/5).
=> STRONG cross-system repeatability of the main reliability-efficiency behaviour; CQR parity on global sharpness with a stability advantage for A3.