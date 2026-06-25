# Report: Final k-NN Experiments

## Objective

This report summarizes the final k-nearest neighbors (k-NN) experiments for the 24-class Greek letter classification task based on precomputed VGG-16 `block4pool` feature vectors. The work was carried out in two consecutive stages.

In the first stage, several k-NN variants were implemented and compared without hyperparameter tuning in order to isolate the effect of the distance metric and voting rule under fixed settings. After this baseline comparison had been completed, a second stage introduced limited hyperparameter tuning to determine whether the observed performance could be improved further.

This two-step procedure makes it possible to distinguish between methodological effects that arise from the classifier design itself and those that result from parameter optimization.

## Data Basis and Evaluation Setup

The experiments are based on:

- `55,329` feature vectors
- `512` features per sample
- `24` target classes

The feature matrix and label vector were loaded from the extracted feature set, and the class mapping was reconstructed from the corresponding JSON file. A stratified `80/20` train-test split was applied in order to preserve the class distribution:

- training set: `44,263` samples
- test set: `11,066` samples

All features were standardized using `StandardScaler`. Since k-NN relies directly on distances in feature space, this preprocessing step is necessary to ensure that all dimensions contribute on a comparable scale.

Performance was evaluated using:

- accuracy
- macro-F1 score
- weighted-F1 score

In addition, normalized confusion matrices were generated for each evaluated model in order to analyze class-specific prediction patterns.

## Stage 1: Comparison Without Hyperparameter Tuning

The first stage focused on direct implementation and comparison of different k-NN formulations under fixed conditions. The objective was to understand how the prediction behavior changes when the distance metric or the voting mechanism is modified, while keeping the remaining experimental setup unchanged.

### 1. Euclidean Majority k-NN

This baseline approach uses Euclidean distance to identify the nearest neighbors and applies unweighted majority voting. Each neighbor contributes equally to the final decision.

Two fixed-neighbor variants were evaluated during this stage:

- `k = 5`
- `k = 7`

**Results for `k = 5`**

- Accuracy: `0.5405`
- Macro-F1: `0.4748`
- Weighted-F1: `0.5349`

![Euclidean majority confusion matrix, k=5](05_final_knn_report_assets/confusion_matrix_euclidean_majority.png)

**Results for `k = 7`**

- Accuracy: `0.5548`
- Macro-F1: `0.4813`
- Weighted-F1: `0.5462`

![Euclidean majority confusion matrix, k=7](05_final_knn_7_report_assets/confusion_matrix_euclidean_majority.png)

### 2. Euclidean Distance-Weighted k-NN

The second approach retains Euclidean distance but replaces majority voting with distance-weighted voting. Neighbor influence is scaled by inverse distance, so closer samples contribute more strongly than more distant ones.

This modification is motivated by the assumption that not all neighbors within the local neighborhood are equally informative.

**Results for `k = 5`**

- Accuracy: `0.5517`
- Macro-F1: `0.4885`
- Weighted-F1: `0.5452`

![Euclidean weighted confusion matrix, k=5](05_final_knn_report_assets/confusion_matrix_euclidean_weighted.png)

**Results for `k = 7`**

- Accuracy: `0.5667`
- Macro-F1: `0.4956`
- Weighted-F1: `0.5581`

![Euclidean weighted confusion matrix, k=7](05_final_knn_7_report_assets/confusion_matrix_euclidean_weighted.png)

### 3. Cosine Distance-Weighted k-NN

The third approach uses L2-normalized feature vectors and retrieves neighbors according to cosine similarity. The similarities are converted into cosine distances, which are then used in a distance-weighted vote.

This changes the underlying notion of similarity: Euclidean distance measures absolute separation in feature space, whereas cosine similarity emphasizes the angular relation between vectors.

**Results for `k = 5`**

- Accuracy: `0.5557`
- Macro-F1: `0.4881`
- Weighted-F1: `0.5480`

![Cosine weighted confusion matrix, k=5](05_final_knn_report_assets/confusion_matrix_cosine_weighted.png)

**Results for `k = 7`**

- Accuracy: `0.5662`
- Macro-F1: `0.4960`
- Weighted-F1: `0.5570`

![Cosine weighted confusion matrix, k=7](05_final_knn_7_report_assets/confusion_matrix_cosine_weighted.png)

## Interpretation of Stage 1

The fixed-parameter comparison leads to three clear observations.

First, distance-weighted voting consistently outperforms unweighted majority voting. This is visible for both neighborhood sizes and indicates that local proximity carries additional class information beyond simple neighbor membership.

Second, increasing the neighborhood size from `k = 5` to `k = 7` improves all three examined variants. This suggests that the smaller neighborhood was slightly too restrictive for the given feature representation.

Third, the difference between Euclidean and cosine distance is small compared with the difference between weighted and unweighted voting. Consequently, the principal methodological gain in the first stage stems from the voting scheme rather than from the metric choice.

Among the untuned models, the strongest class-balanced result is achieved by **Cosine distance-weighted k-NN with `k = 7`** in terms of macro-F1 (`0.4960`), while **Euclidean distance-weighted k-NN with `k = 7`** yields a nearly identical performance profile and a slightly higher weighted-F1.

## Stage 2: Hyperparameter Tuning

After the initial comparison had been completed, a second stage introduced limited hyperparameter tuning in order to test whether the manually specified configurations had already reached a near-optimal operating point.

A `GridSearchCV` procedure with `3`-fold stratified cross-validation was applied. The search was optimized for **macro-F1**, which is appropriate for a multiclass problem with uneven class frequencies because it gives equal importance to each class.

The tuning space included:

- neighbors: `3, 5, 7, 9, 11`
- voting rules: `uniform`, `distance`
- Euclidean and Manhattan distance without additional normalization
- cosine distance with `Normalizer()`

This resulted in `30` candidate configurations and `90` total fits.

### Best Tuned Configuration

The best cross-validation result was:

- Best CV macro-F1: `0.4915`
- Metric: `euclidean`
- Neighbors: `11`
- Weights: `distance`
- Normalizer: `passthrough`

Evaluation on the held-out test set produced:

- Accuracy: `0.5795`
- Macro-F1: `0.5019`
- Weighted-F1: `0.5688`

![Tuned KNN confusion matrix](05_final_knn_7_report_assets/confusion_matrix_tuned_knn.png)

## Comparison Across Both Stages

| Approach | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| Tuned KNN (euclidean, k=11, distance) | 0.5795 | 0.5019 | 0.5688 |
| Cosine distance-weighted KNN, k=7 | 0.5662 | 0.4960 | 0.5570 |
| Euclidean distance-weighted KNN, k=7 | 0.5667 | 0.4956 | 0.5581 |
| Euclidean distance-weighted KNN, k=5 | 0.5517 | 0.4885 | 0.5452 |
| Cosine distance-weighted KNN, k=5 | 0.5557 | 0.4881 | 0.5480 |
| Euclidean majority KNN, k=7 | 0.5548 | 0.4813 | 0.5462 |
| Euclidean majority KNN, k=5 | 0.5405 | 0.4748 | 0.5349 |

## Discussion

The overall development of the work shows a clear progression.

The first stage established a controlled methodological comparison without parameter tuning. This made it possible to identify the most relevant design choices under fixed experimental conditions. The results showed that distance weighting is consistently beneficial and that both Euclidean and cosine distance are competitive when combined with weighted voting.

The second stage then extended this baseline by introducing a small but systematic tuning procedure. This step demonstrated that the manually selected values had not yet exhausted the available performance. In particular, the tuning process identified a larger neighborhood size (`k = 11`) together with Euclidean distance and distance weighting as the strongest overall combination.

Thus, the work was not carried out as a single optimized search from the beginning. Instead, it followed a more structured experimental logic:

1. implement and compare conceptually distinct approaches under fixed settings
2. identify the strongest untuned variants
3. perform limited tuning afterward to quantify the improvement gained through parameter optimization

This sequence is methodologically useful because it separates the effect of model design from the effect of parameter adjustment.

## Final Summary

The experiments show that the most important improvement over the basic k-NN baseline arises from the introduction of distance-weighted voting. A secondary improvement is obtained by increasing the neighborhood size. By contrast, the choice between Euclidean and cosine distance produces only a comparatively small difference when weighted voting is already in place.

The best final result is achieved by the tuned Euclidean distance-weighted model with `k = 11`, which reaches:

- Accuracy: `0.5795`
- Macro-F1: `0.5019`
- Weighted-F1: `0.5688`

From a scientific perspective, the combined analysis indicates that classifier performance in this setting is driven primarily by the weighting strategy and neighborhood size, while the exact distance metric has a smaller effect. The final tuned configuration therefore represents the strongest overall model among the evaluated k-NN variants.
