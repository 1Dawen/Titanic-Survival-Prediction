# Methodology and limitations

## Feature engineering

Passenger titles are extracted from `Name` and normalized, and `FamilySize` and `AgeClass` are derived from the source columns. `IsAlone` is also calculated during exploratory feature engineering but is not included in the final eight-feature model. The categorical model inputs are `Sex`, `Embarked`, and `Title`.

Missing `Age` values are filled from median ages for each title, computed **only on the current training partition**. A training-median fallback handles titles absent from that partition. Missing `Fare` values use the current training partition's median. Age and fare imputation are reused without refitting on validation/test records.

## Validation strategy

The scripts compare CatBoost parameters with `GroupKFold(n_splits=5)` and `Ticket` as the grouping key. Grouping helps prevent passengers on the same ticket from being split across training and validation folds. The group key is not fed to CatBoost. This approach does not guarantee target stratification or guard against every possible family-level relationship.

The recorded configuration comparison covers eight CatBoost combinations of depth, iteration count and learning rate. The selected default model uses depth 5, 500 iterations, learning rate 0.02, and `Logloss`.

## Scope of the reported numbers

The repository retains `results/recorded_cv_results.csv` as a record of model experimentation. Its best mean five-fold accuracy was 0.824832, with a fold standard deviation of 0.046017. These results were saved from the initial Kaggle exploration and have not been rerun on official competition data in this repository release.

The maintained workflow makes the feature set consistent between cross-validation and final fitting, including `Fare`, and handles missing values within each training fold. Results from rerunning it may differ from the saved experiment record. Those differences must not be reported as reproduced accuracy without executing the official data.

The tests use fictional passenger records to check deterministic transformations, fold separation, pipeline behavior and Kaggle submission columns. The official training data is not redistributed. No Kaggle leaderboard score or external held-out test accuracy is claimed.
