# Presentation: Final k-NN Experiments

## Slide 1 - Objective of the Analysis

- Classification of `24` Greek letter classes
- Based on precomputed VGG-16 features from `block4pool`
- Objective:
  - compare different k-NN variants methodologically
  - then apply targeted hyperparameter tuning
- Focus:
  - distance metric
  - voting strategy
  - effect of `k`

## Slide 2 - Data Basis

- Dataset size:
  - `55,329` samples
  - `512` features per sample
  - `24` classes
- Stratified split:
  - training: `44,263`
  - test: `11,066`
- Evaluation metrics:
  - accuracy
  - macro-F1
  - weighted-F1

## Slide 3 - Loading Data and Reconstructing Classes

- Features and labels are loaded from precomputed `.npy` files
- The class mapping is reconstructed from `class_indices.json`
- This makes metrics and confusion matrices interpretable again

```python
PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "extracted_features_alena"

X = np.load(DATA_DIR / "features_block4pool.npy")
y = np.load(DATA_DIR / "labels_block4pool.npy")

with open(DATA_DIR / "class_indices.json", "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)

idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
class_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]
```

## Slide 4 - Preprocessing and Split

- A stratified `80/20` split preserves the class distribution
- `StandardScaler` is essential for k-NN because distance comparisons depend on feature scale
- Without standardization, individual feature dimensions could dominate

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["random_state"],
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Slide 5 - Stage 1: Comparison Without Tuning

- Three model variants were compared under identical conditions
- Tested variants:
  - Euclidean + majority vote
  - Euclidean + distance-weighted vote
  - Cosine + distance-weighted vote
- Objective:
  - isolate the methodological effect
  - no hyperparameter optimization yet

## Slide 6 - Core Idea of the Custom k-NN Implementation

- Distance computation is performed batchwise with PyTorch
- For Euclidean k-NN, nearest neighbors are selected via `topk(..., largest=False)`
- The final prediction is then made by either majority vote or weighted vote

```python
def predict_knn_euclidean(X_train, y_train, X_test, n_classes, k=5, batch_size=64, weighted=False):
    train = _as_float_tensor(X_train)
    test = _as_float_tensor(X_test)
    y_train_t = torch.from_numpy(y_train.astype(np.int64, copy=False))

    train_t = train.T.contiguous()
    train_norm = (train * train).sum(dim=1)

    for start_idx in range(0, len(test), batch_size):
        end_idx = min(start_idx + batch_size, len(test))
        batch = test[start_idx:end_idx]
        dists_sq = (batch * batch).sum(dim=1, keepdim=True) + train_norm[None, :] - 2 * batch.matmul(train_t)
        nn_dists_sq, nn_idx = torch.topk(dists_sq, k=k, largest=False, dim=1)
```

## Slide 7 - Voting Mechanisms in the Code

- Majority vote:
  - every neighbor contributes equally
- Weighted vote:
  - closer neighbors have stronger influence
- This was the most important methodological improvement in the experiments

```python
def _majority_vote(nn_labels, n_classes):
    y_pred = np.empty(nn_labels.shape[0], dtype=np.int64)
    for row_idx, labels in enumerate(nn_labels):
        counts = torch.bincount(labels.long(), minlength=n_classes)
        y_pred[row_idx] = int(counts.argmax())
    return y_pred


def _weighted_vote(nn_labels, weights, n_classes):
    y_pred = np.empty(nn_labels.shape[0], dtype=np.int64)
    for row_idx, (labels, row_weights) in enumerate(zip(nn_labels, weights)):
        scores = torch.zeros(n_classes, dtype=torch.float64)
        scores.scatter_add_(0, labels.long(), row_weights.double())
        y_pred[row_idx] = int(scores.argmax())
    return y_pred
```

## Slide 8 - Cosine Variant in the Code

- The cosine approach uses L2-normalized vectors
- Similarity is measured by angle rather than absolute distance
- The final decision again uses distance-weighted voting

```python
def predict_knn_cosine_weighted(X_train, y_train, X_test, n_classes, k=5, batch_size=64):
    train = torch.nn.functional.normalize(_as_float_tensor(X_train), p=2, dim=1)
    test = torch.nn.functional.normalize(_as_float_tensor(X_test), p=2, dim=1)

    for start_idx in range(0, len(test), batch_size):
        end_idx = min(start_idx + batch_size, len(test))
        batch = test[start_idx:end_idx]
        similarities = batch.matmul(train.T.contiguous())
        nn_similarities, nn_idx = torch.topk(similarities, k=k, largest=True, dim=1)
        cosine_distances = 1 - nn_similarities
        weights = 1 / (torch.clamp(cosine_distances, min=0) + 1e-12)
```

## Slide 9 - Stage 1 Results for k = 5

- Euclidean majority:
  - accuracy `0.5405`
  - macro-F1 `0.4748`
  - weighted-F1 `0.5349`
- Euclidean weighted:
  - accuracy `0.5517`
  - macro-F1 `0.4885`
  - weighted-F1 `0.5452`
- Cosine weighted:
  - accuracy `0.5557`
  - macro-F1 `0.4881`
  - weighted-F1 `0.5480`

## Slide 10 - Stage 1 Results for k = 7

- Euclidean majority:
  - accuracy `0.5548`
  - macro-F1 `0.4813`
  - weighted-F1 `0.5462`
- Euclidean weighted:
  - accuracy `0.5667`
  - macro-F1 `0.4956`
  - weighted-F1 `0.5581`
- Cosine weighted:
  - accuracy `0.5662`
  - macro-F1 `0.4960`
  - weighted-F1 `0.5570`

## Slide 11 - Interpretation of Stage 1

- Distance-weighted voting consistently outperforms majority voting
- `k = 7` performs better than `k = 5` in all three variants
- The difference between Euclidean and cosine is comparatively small
- The main effect comes from:
  - not the metric itself
  - but the weighting of neighbors

## Slide 12 - Stage 2: Hyperparameter Tuning

- After the controlled comparison, a limited tuning stage was added
- Objective:
  - test whether the manually chosen configurations were already close to optimal
- Procedure:
  - `GridSearchCV`
  - `3`-fold stratified cross-validation
  - optimization for `macro-F1`

## Slide 13 - Tuning Search Space in the Code

- The tuning space included:
  - `k = 3, 5, 7, 9, 11`
  - `uniform` and `distance`
  - `euclidean`, `manhattan`, `cosine`
- For cosine, an additional `Normalizer()` was applied

```python
tuning_pipeline = Pipeline([
    ("normalizer", "passthrough"),
    ("knn", KNeighborsClassifier(algorithm="brute")),
])

param_grid = [
    {
        "normalizer": ["passthrough"],
        "knn__n_neighbors": CONFIG["tuning_neighbors"],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    },
    {
        "normalizer": [Normalizer()],
        "knn__n_neighbors": CONFIG["tuning_neighbors"],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["cosine"],
    },
]
```

## Slide 14 - Grid Search and Best Model

- In total:
  - `30` configurations
  - `90` fits with `3` folds
- Best cross-validation result:
  - macro-F1 `0.4915`
  - metric: `euclidean`
  - weights: `distance`
  - `k = 11`

```python
grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train_scaled, y_train)
y_pred_tuned = grid_search.predict(X_test_scaled)
```

## Slide 15 - Best Final Result on the Test Set

- Tuned KNN (`euclidean`, `k = 11`, `distance`)
- Test metrics:
  - accuracy `0.5795`
  - macro-F1 `0.5019`
  - weighted-F1 `0.5688`
- This is the best overall model among all evaluated variants

## Slide 16 - Overall Comparison of All Models

- Tuned KNN (`euclidean`, `k=11`, `distance`): macro-F1 `0.5019`
- Cosine weighted KNN, `k=7`: macro-F1 `0.4960`
- Euclidean weighted KNN, `k=7`: macro-F1 `0.4956`
- Euclidean weighted KNN, `k=5`: macro-F1 `0.4885`
- Cosine weighted KNN, `k=5`: macro-F1 `0.4881`
- Euclidean majority KNN, `k=7`: macro-F1 `0.4813`
- Euclidean majority KNN, `k=5`: macro-F1 `0.4748`

## Slide 17 - Key Findings

- The largest gain comes from distance-weighted voting
- A second gain comes from a larger `k`
- The distance metric matters, but less than the voting strategy
- Tuning still improves the best untuned model by a measurable margin

## Slide 18 - Conclusion

- The methodological sequence was useful:
  - first compare variants
  - then apply targeted tuning
- Best model:
  - Euclidean distance-weighted k-NN with `k = 11`
- Final takeaway:
  - in this setting, performance is driven mainly by neighbor weighting and the choice of `k`
  - the distance metric itself has a smaller influence
