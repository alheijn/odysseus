# Erklaerung: `05_final_knn_7.ipynb`

## Worum geht es in diesem Notebook?

Dieses Notebook ist eine **erweiterte Version** von `05_final_knn.ipynb`.

Es macht zuerst fast dasselbe:

- Daten laden
- splitten
- standardisieren
- drei manuelle k-NN-Varianten testen

Danach kommt noch ein vierter Schritt:

- **kleines Hyperparameter-Tuning mit `GridSearchCV`**

Das ist der wichtigste Unterschied.

---

## Der Hauptunterschied zu `05_final_knn.ipynb`

Im normalen `05_final_knn` ist `k = 5` fest vorgegeben.

In `05_final_knn_7`:

- startet die manuelle Auswertung mit `k = 7`
- und danach wird zusaetzlich automatisch ausprobiert:
  - verschiedene `k`-Werte
  - verschiedene Distanzmetriken
  - verschiedene Gewichtungen
  - mit oder ohne Normalisierung

Das Ziel ist also nicht nur vergleichen, sondern auch **gezielt bessere Einstellungen finden**.

---

## Zelle 1: Imports

Die meisten Imports sind identisch zum anderen Notebook.
Neu hinzu kommen:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
```

### Wichtige neue Bausteine

- `KNeighborsClassifier`:
  fertige k-NN-Implementierung aus scikit-learn
- `Pipeline`:
  kombiniert Vorverarbeitung und Modell
- `GridSearchCV`:
  probiert viele Parameterkombinationen systematisch aus
- `StratifiedKFold`:
  macht Kreuzvalidierung mit erhaltener Klassenverteilung
- `Normalizer`:
  skaliert jeden einzelnen Vektor auf Laenge 1

---

## Zelle 2: Konfiguration

```python
CONFIG = {
    "n_neighbors": 7,
    "batch_size": 64,
    "test_size": 0.20,
    "random_state": 42,
    "tuning_neighbors": [3, 5, 7, 9, 11],
    "tuning_cv_folds": 3,
}
```

### Unterschiede zu vorher

- das manuelle k-NN nutzt hier `k = 7`
- fuer das Tuning werden getestet:
  - `3, 5, 7, 9, 11`
- die Kreuzvalidierung nutzt `3` Folds

---

## Zellen 3 und 4: Daten laden, Split, Standardisierung

Dieser Teil ist inhaltlich fast gleich wie im anderen Notebook.

Geladen werden:

- `features_block4pool.npy`
- `labels_block4pool.npy`
- `class_indices.json`

Die Shapes laut Output:

- `X.shape = (55329, 512)`
- `y.shape = (55329,)`
- `n_classes = 24`

Train/Test:

- Train: `(44263, 512)`
- Test: `(11066, 512)`

Dann:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Auch hier ist das fuer distanzbasierte Verfahren sehr wichtig.

---

## Zelle 6: Hilfsfunktionen

Die Hilfsfunktionen sind praktisch dieselben wie in `05_final_knn.ipynb`:

- `_as_float_tensor`
- `_majority_vote`
- `_weighted_vote`
- `predict_knn_euclidean`
- `predict_knn_cosine_weighted`
- `evaluate_approach`
- `plot_confusion_matrix`

### Bedeutet konkret

Die ersten drei Ansätze werden wieder **manuell mit Torch** berechnet und nicht mit `KNeighborsClassifier`.

Das ist interessant, weil man hier den Algorithmus viel besser versteht:

- Distanz berechnen
- Nachbarn auswaehlen
- Abstimmung machen

---

## Zellen 8, 10 und 12: Die drei manuellen k-NN-Varianten

Dieselben Ideen wie im anderen Notebook, nur mit `k = 7`.

---

### 1. Euclidean majority KNN

```python
y_pred_euclidean_majority = predict_knn_euclidean(... weighted=False)
```

Ergebnis laut Output:

- Accuracy: `0.5548`
- Macro-F1: `0.4813`
- Weighted-F1: `0.5462`

Gegenueber dem anderen Notebook ist das etwas besser als bei `k = 5`.

---

### 2. Euclidean distance-weighted KNN

```python
y_pred_euclidean_weighted = predict_knn_euclidean(... weighted=True)
```

Ergebnis laut Vergleichstabelle:

- Accuracy: `0.5667`
- Macro-F1: `0.4956`
- Weighted-F1: `0.5581`

Diese Variante ist wieder besser als die einfache Mehrheitsabstimmung.

---

### 3. Cosine distance-weighted KNN

```python
y_pred_cosine_weighted = predict_knn_cosine_weighted(...)
```

Ergebnis:

- Accuracy: `0.5662`
- Macro-F1: `0.4960`
- Weighted-F1: `0.5570`

Auch hier sehr stark, und beim `macro_f1` sogar minimal besser als die euklidisch gewichtete Version mit `k = 7`.

---

## Zelle 14: Kleines Hyperparameter-Tuning

Das ist der wichtigste neue Teil.

### 1. Pipeline aufbauen

```python
tuning_pipeline = Pipeline([
    ("normalizer", "passthrough"),
    ("knn", KNeighborsClassifier(algorithm="brute")),
])
```

Diese Pipeline hat zwei Schritte:

1. `normalizer`
2. `knn`

`"passthrough"` bedeutet:
Im ersten Fall wird gar keine zusaetzliche Normalisierung gemacht.

Warum eine Pipeline?

Damit `GridSearchCV` verschiedene Kombinationen sauber testen kann.

---

### 2. Parameterraum definieren

```python
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

Hier wird festgelegt, was ausprobiert werden soll.

### Fall A: ohne zusaetzliche Normalisierung

Es werden kombiniert:

- `n_neighbors`: 3, 5, 7, 9, 11
- `weights`: `uniform` oder `distance`
- `metric`: `euclidean` oder `manhattan`

### Fall B: mit `Normalizer()`

Hier wird nur `cosine` getestet.

Warum?

Kosinus-Vergleiche passen in der Praxis gut zu normierten Vektoren.

---

## Was bedeuten diese Parameter?

### `n_neighbors`

Wie viele Nachbarn abstimmen.

### `weights`

- `uniform`: alle gleich wichtig
- `distance`: naeher gelegene Nachbarn zaehlen staerker

### `metric`

- `euclidean`: Luftlinienabstand
- `manhattan`: Summe absoluter Unterschiede
- `cosine`: vergleicht eher Richtung als Laenge

### `Normalizer()`

Normiert jeden einzelnen Feature-Vektor auf Laenge 1.
Das ist etwas anderes als `StandardScaler`.

- `StandardScaler`: arbeitet spaltenweise pro Feature
- `Normalizer`: arbeitet zeilenweise pro Beispiel

Das Notebook benutzt also:

1. zuerst `StandardScaler`
2. und im Cosine-Tuning optional zusaetzlich `Normalizer`

---

## Zelle 14: Kreuzvalidierung definieren

```python
cv = StratifiedKFold(
    n_splits=CONFIG["tuning_cv_folds"],
    shuffle=True,
    random_state=CONFIG["random_state"],
)
```

Das bedeutet:

- 3-fache Kreuzvalidierung
- Klassenverteilung bleibt in jedem Fold aehnlich
- Mischen mit festem Seed

### Kurz erklaert

Beim Tuning wird das Training mehrfach in:

- Teil zum Lernen
- Teil zum Validieren

aufgeteilt.

So kann man Parameter fairer vergleichen als mit nur einem einzigen Split.

---

## Zelle 14: `GridSearchCV`

```python
grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)
```

### Bedeutende Optionen

- `estimator=tuning_pipeline`:
  die Pipeline soll optimiert werden
- `param_grid=param_grid`:
  diese Kombinationen sollen getestet werden
- `scoring="f1_macro"`:
  die Auswahl richtet sich nach Macro-F1
- `cv=cv`:
  3-fache Kreuzvalidierung
- `n_jobs=-1`:
  alle verfuegbaren CPU-Kerne nutzen

Dann:

```python
grid_search.fit(X_train_scaled, y_train)
```

Jetzt probiert scikit-learn alle Kombinationen durch.

---

## Bestes Ergebnis aus dem Tuning

Nach dem Training:

```python
best_params = grid_search.best_params_.copy()
```

Der beste Ansatz laut Notebook ist:

- `metric = euclidean`
- `n_neighbors = 11`
- `weights = distance`
- `normalizer = passthrough`

Und der beste CV-Wert ist:

- `cv_macro_f1 = 0.491515`

Danach wird mit diesem besten Modell auf dem echten Testset vorhergesagt:

```python
y_pred_tuned = grid_search.predict(X_test_scaled)
```

---

## Warum ist der Testwert hoeher als der CV-Wert?

In der Tabelle steht:

- `cv_macro_f1 = 0.4915`
- Test-`macro_f1 = 0.5019`

Das ist nicht falsch.

Kreuzvalidierung und Testset sind verschiedene Auswertungen.
Der Testwert kann etwas hoeher oder etwas niedriger sein.

Wichtig ist nur:

- getuned wurde auf dem Training
- bewertet wurde danach auf dem separaten Testset

Das ist methodisch sauber.

---

## Ergebnis des getunten Modells

Laut Vergleichstabelle:

- Accuracy: `0.5795`
- Macro-F1: `0.5019`
- Weighted-F1: `0.5688`

Das ist besser als alle drei manuellen Varianten.

---

## Zelle 16: Finale Vergleichstabelle

```python
results_df = pd.DataFrame([
    metrics_euclidean_majority,
    metrics_euclidean_weighted,
    metrics_cosine_weighted,
    metrics_tuned,
]).sort_values("macro_f1", ascending=False).reset_index(drop=True)
```

Die vier verglichenen Ansätze sind:

1. Tuned KNN `(euclidean, k=11, distance)`
2. Cosine distance-weighted KNN
3. Euclidean distance-weighted KNN
4. Euclidean majority KNN

Sortiert wird wieder nach `macro_f1`.

---

## Zelle 17: Beste Methode ausgeben

Das Notebook meldet als beste Methode:

```text
Tuned KNN (euclidean, k=11, distance)
```

Mit:

- Accuracy: `0.5795`
- Macro-F1: `0.5019`
- Weighted-F1: `0.5688`

---

## Was lernt man aus diesem Notebook?

### 1. `k = 7` ist schon besser als `k = 5`

Die manuellen Varianten schneiden hier etwas besser ab als im anderen Notebook.

### 2. Gewichtete Abstimmung ist wieder sinnvoll

Distanzgewichtung hilft auch hier.

### 3. Automatisches Tuning bringt noch mehr

Das Grid Search findet eine noch bessere Kombination:

- mehr Nachbarn (`k = 11`)
- euklidische Distanz
- Distanzgewichtung

### 4. Macro-F1 bleibt der zentrale Massstab

Nicht Accuracy entscheidet, sondern wieder `macro_f1`.
Das ist bei vielen Klassen und moeglicher Unbalance oft die bessere Wahl.

---

## Unterschied zwischen den drei Ansätzen und dem getunten Modell

Die ersten drei Varianten sind eher:

- handgebaut
- zum Verstehen des Algorithmus
- zum gezielten Vergleichen einzelner Ideen

Das getunte Modell ist eher:

- systematischer
- naeher an einer echten Modellauswahl
- praktischer fuer die finale Entscheidung

---

## Die Hauptidee in einem Satz

`05_final_knn_7.ipynb` vergleicht erst mehrere manuelle k-NN-Varianten mit `k = 7` und findet danach per Grid Search ein noch besseres Modell: euklidisches k-NN mit Distanzgewichtung und `k = 11`.

---

## Kurzfassung fuer Anfaenger

- Lade vorberechnete Features.
- Teile sie in Training und Test.
- Standardisiere die Werte.
- Teste drei k-NN-Varianten manuell.
- Lass danach `GridSearchCV` viele Parameterkombinationen ausprobieren.
- Vergleiche alles mit `macro_f1`.
- Nimm das beste Modell.

