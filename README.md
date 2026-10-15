# Flight Price Prediction Using Machine Learning

## 1. Problem Statement
The objective is to predict domestic flight ticket prices using flight attributes such as airline, route, departure and arrival times, stops, flight class, duration, and days left before departure. This is a supervised regression problem with `price` as the target variable.

## 2. Dataset and Preprocessing
- Training dataset: 18,076 rows and 12 columns.
- Test dataset: 4,520 rows and 11 columns.
- No missing values or duplicate records were detected.
- The `id` column was excluded from model training but retained for submission.
- Categorical features were one-hot encoded for XGBoost, while CatBoost used native categorical-feature handling.
- An 80:20 train-validation split with `random_state=42` was used for evaluation.

## 3. Models Tried
- **Baseline XGBoost:** Established the initial performance benchmark.
- **Tuned XGBoost:** Randomized hyperparameter search with three-fold cross-validation.
- **CatBoost:** Evaluated native categorical-feature handling.
- **Weighted Ensemble:** Combined CatBoost and XGBoost predictions using different weights.

## 4. Validation Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Baseline XGBoost | 50.9015 | 146.3078 | 0.6195 |
| Tuned XGBoost | 50.4377 | 145.9550 | 0.6213 |
| CatBoost | 49.0015 | 141.0526 | 0.6463 |
| Ensemble (30% CatBoost + 70% XGBoost) | 43.0040 | 112.9787 | 0.7731 |
| **XGBoost (best reported result)** | **42.0054** | **103.6396** | **0.8091** |

*Note: The best XGBoost result differs from the earlier tuned-model result and should be verified against the notebook before final reporting.*

## 5. Results and Conclusion
CatBoost outperformed the initial XGBoost models, while the weighted ensemble improved upon CatBoost but did not outperform XGBoost in the latest comparison. XGBoost is therefore the current preferred candidate based on validation performance.

The selected model is retrained on the full training dataset to predict prices for the test dataset. The final Kaggle submission contains `id` and predicted `price` columns, with 4,520 rows. Trained models and preprocessing pipelines are saved for reuse.