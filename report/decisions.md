# Flight Price Prediction — Decision Log

## EDA Phase

### Data Cleaning

| Decision | Choice | Reason |
|---|---|---|
| `price <= 0` rows | Drop (25 rows) | All are Southwest Economy one-stop flights — scraping artifact or companion fare, not real market prices. These 25 rows alone shift log1p(price) skew from −0.03 to −0.74, distorting the target distribution. |
| `Southern_Airways_Express` | Drop the single training row | 1 training example, 0 test rows. A one-hot column learned from one row carries no generalizable signal. |

### Features Dropped

| Feature | Reason |
|---|---|
| `id` | Row identifier with no predictive signal |
| `flight` | 1,123 unique codes. The flight-code prefix maps 1-to-1 onto airline (AA → American, WN → Southwest, etc.) so the column adds no information beyond what `airline` already captures. OHE would also inflate feature space. |

### Encoding Decisions

| Feature | Encoding | Reason |
|---|---|---|
| `stops` | Ordinal: zero=0, one=1, two_or_more=2 | Naturally ordered; a single integer preserves ordering without extra columns |
| `flight_class` | Binary: Economy=0, Business=1 | Two values with clear ordinal meaning |
| `departure_time`, `arrival_time` | Ordinal: Early_Morning=0, Morning=1, Afternoon=2, Evening=3, Night=4, Late_Night=5 | Time-of-day has a natural order; ordinal encoding is more compact than OHE. `Late_Night` has only 6 training rows — effect will be noisy but no new column is created |
| `airline`, `source_city`, `destination_city` | One-hot encode | No natural order; cardinality is small enough that OHE is practical |

### Feature Engineering

| Feature | Decision | Reason |
|---|---|---|
| `last_week` | Add binary flag: `days_left <= 7` | Median price is flat for the final week (~$305–315) then falls steadily to ~$179 at 31–44 days out. A linear model fits a slope to the whole range and misses this kink. The flag is cheap to add and simple to justify. |

### Target Transformation

| Decision | Choice | Reason |
|---|---|---|
| `price` transform | `log1p(price)` | Raw price is right-skewed. After cleaning, log1p(price) is nearly symmetric (skew ≈ −0.03). Predictions are converted back via `expm1` at inference time. High-end Business fares (max ~$8,996) are plausibly real and kept — the log transform already reduces their influence. |

### Feature Scaling

| Decision | Choice | Reason |
|---|---|---|
| Input scaling | `StandardScaler` (zero mean, unit variance) on all numeric inputs | Ridge, Lasso, and ElasticNet apply an L1/L2 penalty to coefficients. Without scaling, features with larger magnitudes are penalised more heavily regardless of importance. |

### Domain Notes (for viva)

- **`class_num` r ≈ 0.42**: This is correct for US domestic first-class, which carries a ~1.7× premium over Economy. Other datasets (e.g. Indian Kaggle flight data) show r ≈ 0.9 because long-haul business class is ~7× more expensive. The encoding is right; the dataset simply behaves differently.
- **`flight` prefix → airline**: Every flight code in the dataset begins with the airline's IATA code, so `flight` is a strict superset of `airline` and adds nothing after `airline` is already encoded.

---

## Modeling Phase

### Preprocessing Pipeline

scikit-learn `Pipeline` + `ColumnTransformer` — transformations fit only on the training fold inside cross-validation, preventing leakage. Viva answer: "the scaler never saw the validation fold."

| Column group | Transformer |
|---|---|
| `duration`, `days_left`, ordinal-encoded cols | StandardScaler |
| `last_week` | StandardScaler (binary, but scaling is harmless) |
| `airline`, `source_city`, `destination_city` | OneHotEncoder(drop='first', handle_unknown='ignore') |
| `id`, `flight` | dropped |

### Linear Model Results (02_modeling.ipynb)

5-fold CV RMSE in dollars. All models use the same KFold(n_splits=5, shuffle=True, random_state=42).

| Rank | Model | CV RMSE ($) |
|---|---|---|
| 1 | M2 LinearRegression — all features, raw target | 166.75 |
| 2 | M3 LinearRegression — log target + last_week | 167.01 |
| 3 | M4 Ridge — alpha=4.41 | 167.02 |
| 4 | M5b ElasticNet — alpha=0.0100, l1=0.2 | 167.80 |
| 5 | M5a Lasso — alpha=0.0100 | 170.25 |
| 6 | M1 LinearRegression — numeric only (baseline) | 207.00 |

**Key observations:**
- M1→M2 (+$40): categorical features carry most of the signal, especially `flight_class` and `airline`
- M2→M3 (+$0.26): log transform and `last_week` flag add negligible improvement — linear model already fits the relationship adequately on raw price
- M3→M4 (~$0): regularisation gives no meaningful gain — multicollinearity is not a significant problem in this dataset
- Lasso worsened RMSE despite zeroing some features, suggesting most features carry genuine signal

---

## Beyond Linear Models

Random forest explored in `03_random_forest.ipynb`. Model constraint from the course (LinearRegression / Ridge / Lasso / ElasticNet only) applies to the assessed submission; RF is exploratory and will be studied and explained independently for the viva.

### Random Forest Results (03_random_forest.ipynb)

| Rank | Model | CV RMSE ($) |
|---|---|---|
| 1 | RF tuned (n_estimators=100, min_samples_leaf=2, max_features=0.5, max_depth=20) | 122.60 |
| 2 | RF baseline (sklearn defaults) | 129.03 |
| 3 | M2 LinearRegression all features | 166.75 |
| — | M3–M5 Ridge/Lasso/ElasticNet | 167–170 |
| 8 | M1 LinearRegression numeric only | 207.00 |

RF tuned beats the best linear model by **$44.15 (26% lower RMSE)**. Selected as the final submission model.

**Key observations:**
- RF baseline already beats all linear models without any tuning — the non-linear relationships (days_left curve, feature interactions) account for most of the gain
- Tuning adds another $6 on top; `max_features=0.5` (using half the features per split) and `min_samples_leaf=2` (slight regularisation) are the meaningful parameters
- `max_depth=20` vs `None` makes little difference at this data size

### Random Forest — design decisions

- **No StandardScaler**: RF is scale-invariant (splits on rank, not magnitude); numeric and ordinal columns passed through unchanged
- **OHE retained**: RF handles integers fine, but OHE is kept for the categorical columns to avoid implying false ordinal relationships (e.g. airline A < airline B)
- **Target**: raw price (no log transform needed — RF has no linearity assumption, and raw target makes the RMSE directly comparable to M2)
- **Tuning**: RandomizedSearchCV over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`

### Extended Experiments (04_gradient_boosting, 05_log_target)

All further experiments plateaued at ~$121 RMSE regardless of model or target transform:

| Experiment | CV RMSE ($) |
|---|---|
| GB tuned, raw target | ~121 |
| RF tuned, log target | ~121 |
| GB tuned, log target | ~121 |

**Conclusion:** $121 is likely the irreducible noise floor — dynamic pricing variance and Business fare outliers that the available features cannot explain. RF tuned ($122.60) selected as final submission model. Further model complexity yields no meaningful gain.
