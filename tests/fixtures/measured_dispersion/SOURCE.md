# Measured dispersion table (test input)

`dispersion_from_fft_ridge_smooth_b.csv` is the measured Bloch dispersion of the fabricated SMV1245 line. It
is extracted from the near-field FFT ridge and smoothed. Tier: `MEASURED`.

| Field | Value |
|---|---|
| Columns | `Frequency_GHz`, `beta_norm` (= beta p / pi) |
| Rows | 400 points, 2.30-20.0 GHz |
| SHA-256 | `f06d016c772b8800d1fa5fe2edbcbbf79496feeab00d8592ddb2d5cb18d6c4b1` |
| Data-deposit copies | `measurements/Fig_dispersion_Gain/dispersion_from_fft_ridge_smooth_b.csv` and `runs/r3_via_fit/inputs/dispersion_from_fft_ridge_smooth_b.csv` |

`tests/test_cascade_budget.py` uses it to check the measured-dispersion exceptional point at 3.1 GHz:
`Delta_theta_meas = 24.75 deg`, `m_EP = 3.99` with the model `kappa/m`.
