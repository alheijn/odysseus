# Erklaerung: `05_final_knn.ipynb`

## Worum geht es in diesem Notebook?

Dieses Notebook testet mehrere **k-NN-Varianten** auf bereits berechneten Bild-Features.

Die Bilder selbst werden hier nicht mehr direkt verarbeitet.
Stattdessen arbeitet das Notebook mit den gespeicherten Feature-Vektoren aus einem frueheren Schritt.

Ziel:

- Daten laden
- in Training und Test splitten
- Features standardisieren
- mehrere k-NN-Ansätze vergleichen
- Metriken und Konfusionsmatrix ausgeben
- die beste Variante bestimmen

---

## Was ist k-NN?

k-NN steht fuer **k-nearest neighbors**.

Die Idee:

1. Fuer ein neues Testbeispiel sucht man die `k` aehnlichsten Trainingsbeispiele.
2. Diese Nachbarn "stimmen ab".
3. Die haeufigste Klasse gewinnt.

Hier ist `k = 5`.

Das Notebook testet drei Varianten:

1. euklidische Distanz + einfache Mehrheitsabstimmung
2. euklidische Distanz + gewichtete Abstimmung
3. Kosinus-Aehnlichkeit + gewichtete Abstimmung

---

## Zelle 1: Imports

```python
from pathlib import Path
import json
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

### Wichtige Bibliotheken

- `numpy`: Datenarrays
- `pandas`: Tabellen fuer den Vergleich
- `torch`: hier nicht fuer Training, sondern fuer schnelle Tensor-Rechnungen
- `sklearn.metrics`: Auswertungsmetriken
- `train_test_split`: Daten in Trainings- und Testmenge aufteilen
- `StandardScaler`: Standardisierung der Features
- `matplotlib` und `seaborn`: Visualisierung

Hinweis:
`wandb` wird importiert, im gezeigten Notebook aber nicht weiter benutzt.

---

## Zelle 2: Konfiguration

```python
CONFIG = {
    "n_neighbors": 5,
    "batch_size": 64,
    "test_size": 0.20,
    "random_state": 42,
}
```

Das sind die wichtigsten Einstellungen:

- `n_neighbors = 5`:
  es werden 5 Nachbarn verwendet
- `batch_size = 64`:
  Vorhersagen werden in Paketen berechnet
- `test_size = 0.20`:
  20 % der Daten gehen in den Test
- `random_state = 42`:
  sorgt fuer reproduzierbare Ergebnisse

---

## Zelle 3: Projektordner finden und Daten laden

### Funktion `find_project_root`

```python
def find_project_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "data" / "extracted_features_alena").exists():
            return path
```

Diese Funktion sucht vom aktuellen Ordner aus nach oben, bis sie den Projektordner findet.

Gesucht wird der Ordner:

```python
data/extracted_features_alena
```

Das macht das Notebook robuster, weil es nicht an genau einem Startpfad haengt.

### Daten laden

```python
X = np.load(DATA_DIR / "features_block4pool.npy")
y = np.load(DATA_DIR / "labels_block4pool.npy")
```

- `X`: die Feature-Vektoren
- `y`: die Klassenlabels

Laut Output:

- `X.shape = (55329, 512)`
- `y.shape = (55329,)`

Das bedeutet:

- 55.329 Beispiele
- pro Beispiel 512 Merkmale

### Klassennamen laden

```python
with open(DATA_DIR / "class_indices.json", "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)
```

Hier wird ein Dictionary geladen, das z. B. Klassenname und Index verknuepft.

Danach:

```python
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
class_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]
n_classes = len(class_names)
```

Damit wird:

- aus Zahl -> Klassenname
- eine sortierte Liste aller Klassen
- `n_classes = 24`

Diese Namen werden spaeter fuer den Report und die Konfusionsmatrix gebraucht.

---

## Zelle 4: Train-Test-Split und Standardisierung

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["random_state"],
    stratify=y,
)
```

### Was macht `train_test_split`?

Die Daten werden in zwei Teile getrennt:

- Training
- Test

Mit `stratify=y` wird darauf geachtet, dass die Klassenverteilung in beiden Teilen aehnlich bleibt.

Laut Output:

- Train: `(44263, 512)`
- Test: `(11066, 512)`

### Standardisierung

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Das ist ein sehr wichtiger Schritt.

`StandardScaler` sorgt dafuer, dass jedes Feature ungefaehr:

- Mittelwert 0
- Standardabweichung 1

bekommt.

Warum ist das fuer k-NN wichtig?

Weil k-NN auf Distanzen basiert.
Wenn einzelne Merkmale viel groessere Zahlenbereiche haben als andere, dominieren sie die Distanz.

Wichtig:

- `fit_transform` nur auf dem Training
- `transform` auf dem Test

So verhindert man Data Leakage.

---

## Zelle 6: Hilfsfunktionen

Diese Zelle ist der Kern des Notebooks.

### `_as_float_tensor`

```python
def _as_float_tensor(array):
    return torch.from_numpy(array.astype(np.float32, copy=False))
```

Diese Funktion wandelt ein NumPy-Array in einen PyTorch-Tensor um.

Warum?
Weil die Distanzberechnungen mit Torch hier effizient gemacht werden.

---

### `_majority_vote`

```python
def _majority_vote(nn_labels, n_classes):
    y_pred = np.empty(nn_labels.shape[0], dtype=np.int64)
    for row_idx, labels in enumerate(nn_labels):
        counts = torch.bincount(labels.long(), minlength=n_classes)
        y_pred[row_idx] = int(counts.argmax())
    return y_pred
```

Diese Funktion macht die klassische k-NN-Abstimmung.

#### Eingabe:

- `nn_labels`: fuer jedes Testbeispiel die Labels der `k` Nachbarn

#### Ablauf:

1. `torch.bincount(...)` zaehlt, wie oft jede Klasse vorkommt
2. `argmax()` nimmt die Klasse mit den meisten Stimmen

Das ist die normale Mehrheitsentscheidung.

---

### `_weighted_vote`

```python
def _weighted_vote(nn_labels, weights, n_classes):
    y_pred = np.empty(nn_labels.shape[0], dtype=np.int64)
    for row_idx, (labels, row_weights) in enumerate(zip(nn_labels, weights)):
        scores = torch.zeros(n_classes, dtype=torch.float64)
        scores.scatter_add_(0, labels.long(), row_weights.double())
        y_pred[row_idx] = int(scores.argmax())
    return y_pred
```

Hier stimmen die Nachbarn nicht alle gleich stark ab.

Nachbarn mit hoeherem Gewicht zaehlen mehr.

#### Wichtig:

`scatter_add_` addiert die Gewichte zu der passenden Klasse.

Beispiel:

- Klasse A kommt 2-mal vor, aber mit hohen Gewichten
- Klasse B kommt 3-mal vor, aber mit kleinen Gewichten

Dann kann trotzdem Klasse A gewinnen.

---

## `predict_knn_euclidean`

```python
def predict_knn_euclidean(..., weighted=False):
```

Diese Funktion berechnet k-NN mit **euklidischer Distanz**.

### Schritt fuer Schritt

#### 1. Daten in Tensoren umwandeln

```python
train = _as_float_tensor(X_train)
test = _as_float_tensor(X_test)
y_train_t = torch.from_numpy(y_train.astype(np.int64, copy=False))
```

#### 2. Hilfswerte vorbereiten

```python
train_t = train.T.contiguous()
train_norm = (train * train).sum(dim=1)
```

Hier wird eine schnellere Distanzberechnung vorbereitet.

#### 3. Batchweise ueber Testdaten laufen

```python
for start_idx in range(0, len(test), batch_size):
```

Nicht alle Testdaten auf einmal, sondern stueckweise.
Das spart Speicher.

#### 4. Distanzformel

```python
dists_sq = (batch * batch).sum(dim=1, keepdim=True) + train_norm[None, :] - 2 * batch.matmul(train_t)
```

Das berechnet die **quadrierte euklidische Distanz** zwischen jedem Testpunkt im Batch und allen Trainingspunkten.

Man kann sich das so merken:

- je kleiner der Wert
- desto aehnlicher sind sich zwei Beispiele

#### 5. Naechste Nachbarn finden

```python
nn_dists_sq, nn_idx = torch.topk(dists_sq, k=k, largest=False, dim=1)
nn_labels = y_train_t[nn_idx]
```

- `topk(..., largest=False)` nimmt die kleinsten Distanzen
- `nn_idx` sind die Positionen der Nachbarn
- `nn_labels` sind deren Klassen

#### 6. Abstimmen

Wenn `weighted=False`:

```python
batch_pred = _majority_vote(nn_labels, n_classes)
```

Wenn `weighted=True`:

```python
weights = 1 / (torch.sqrt(torch.clamp(nn_dists_sq, min=0)) + 1e-12)
batch_pred = _weighted_vote(nn_labels, weights, n_classes)
```

Hier gilt:

- kleine Distanz -> grosses Gewicht
- grosse Distanz -> kleines Gewicht

`1e-12` verhindert Division durch 0.

---

## `predict_knn_cosine_weighted`

```python
def predict_knn_cosine_weighted(...):
```

Diese Funktion nutzt **Kosinus-Aehnlichkeit** statt euklidischer Distanz.

### Wichtiger Unterschied

Vorher:

```python
train = torch.nn.functional.normalize(..., p=2, dim=1)
test = torch.nn.functional.normalize(..., p=2, dim=1)
```

Hier werden alle Vektoren auf Laenge 1 normiert.

Dann:

```python
similarities = batch.matmul(train_t)
```

Da normierte Vektoren verwendet werden, entspricht das Skalarprodukt direkt der Kosinus-Aehnlichkeit.

- groesser = aehnlicher
- kleiner = unähnlicher

Deshalb:

```python
nn_similarities, nn_idx = torch.topk(similarities, k=k, largest=True, dim=1)
```

Hier werden die **groessten** Werte genommen.

Danach:

```python
cosine_distances = 1 - nn_similarities
weights = 1 / (torch.clamp(cosine_distances, min=0) + 1e-12)
```

Auch hier wieder:

- kleinere Distanz -> groesseres Gewicht

---

## `evaluate_approach`

```python
def evaluate_approach(name, y_true, y_pred):
```

Diese Funktion berechnet die wichtigsten Kennzahlen:

- `accuracy`
- `macro_f1`
- `weighted_f1`

### Warum drei Metriken?

#### Accuracy

Anteil korrekt klassifizierter Beispiele.

#### Macro-F1

F1 pro Klasse berechnen und dann mitteln.
Alle Klassen zaehlen gleich stark.

Das ist bei unbalancierten Klassen oft wichtiger als Accuracy.

#### Weighted-F1

Auch F1 pro Klasse, aber Klassen mit mehr Beispielen bekommen mehr Gewicht.

Zusätzlich wird ein `classification_report` ausgegeben.
Der zeigt pro Klasse:

- Precision
- Recall
- F1-Score
- Support

---

## `plot_confusion_matrix`

```python
cm = confusion_matrix(y_true, y_pred, labels=sorted(idx_to_class), normalize="true")
```

Die Konfusionsmatrix zeigt:

- welche Klassen gut erkannt werden
- welche Klassen oft verwechselt werden

`normalize="true"` bedeutet:
Jede Zeile wird auf 1 normiert.

Also liest man:

- Zeile = echte Klasse
- Spalte = vorhergesagte Klasse
- hohe Werte auf der Diagonale = gut

Das Heatmap-Bild wird gespeichert als:

```python
knn_confusion_matrix.png
```

---

## Zellen 8, 10 und 12: Die drei getesteten Ansaetze

### 1. Euclidean majority KNN

```python
y_pred_euclidean_majority = predict_knn_euclidean(... weighted=False)
```

Das ist klassisches k-NN:

- euklidische Distanz
- 5 Nachbarn
- jeder Nachbar zaehlt gleich

Ergebnis laut Notebook:

- Accuracy: `0.5405`
- Macro-F1: `0.4748`
- Weighted-F1: `0.5349`

---

### 2. Euclidean distance-weighted KNN

```python
y_pred_euclidean_weighted = predict_knn_euclidean(... weighted=True)
```

Hier:

- euklidische Distanz
- 5 Nachbarn
- naeher gelegene Nachbarn zaehlen staerker

Ergebnis laut Vergleichstabelle:

- Accuracy: `0.5517`
- Macro-F1: `0.4885`
- Weighted-F1: `0.5452`

Das ist laut `macro_f1` die beste Variante in diesem Notebook.

---

### 3. Cosine distance-weighted KNN

```python
y_pred_cosine_weighted = predict_knn_cosine_weighted(...)
```

Hier:

- Kosinus-Aehnlichkeit
- 5 Nachbarn
- gewichtete Abstimmung

Ergebnis:

- Accuracy: `0.5557`
- Macro-F1: `0.4881`
- Weighted-F1: `0.5480`

Interessant:

- Accuracy ist hier sogar etwas hoeher
- aber `macro_f1` ist minimal schlechter als bei der euklidisch gewichteten Variante

Darum gewinnt im Notebook die euklidisch gewichtete Version.

---

## Zelle 14: Finale Vergleichstabelle

```python
results_df = pd.DataFrame([
    metrics_euclidean_majority,
    metrics_euclidean_weighted,
    metrics_cosine_weighted,
]).sort_values("macro_f1", ascending=False).reset_index(drop=True)
```

Hier werden alle Ergebnisse in eine Tabelle gepackt und nach `macro_f1` sortiert.

Warum nach `macro_f1`?

Weil diese Metrik fairer zu kleineren Klassen ist.

Die Tabelle zeigt:

1. Euclidean distance-weighted KNN
2. Cosine distance-weighted KNN
3. Euclidean majority KNN

---

## Zelle 15: Beste Methode ausgeben

```python
best = results_df.iloc[0]
```

Dann werden die Werte des besten Eintrags ausgegeben:

- bester Ansatz
- Accuracy
- Macro-F1
- Weighted-F1

Im Notebook ist das:

```text
Euclidean distance-weighted KNN
```

---

## Was lernt man aus diesem Notebook?

### 1. Standardisierung ist wichtig

Vor dem Distanzvergleich muessen die Features skaliert werden.

### 2. Gewichtete Nachbarn helfen

Die gewichtete euklidische Variante ist besser als einfache Mehrheitsabstimmung.

### 3. Accuracy allein reicht nicht

Obwohl die Kosinus-Variante bei Accuracy leicht besser ist, gewinnt sie nicht nach `macro_f1`.

Das zeigt:
Welche Metrik man optimiert, macht einen Unterschied.

---

## Die Hauptidee in einem Satz

`05_final_knn.ipynb` vergleicht drei k-NN-Strategien auf 512-dimensionalen Bild-Features und entscheidet anhand von `macro_f1`, dass die euklidisch gewichtete Variante die beste ist.

---

## Kurzfassung fuer Anfaenger

- Lade vorberechnete Bild-Features.
- Teile sie in Training und Test.
- Standardisiere die Werte.
- Suche fuer jedes Testbeispiel die 5 naechsten Nachbarn.
- Lasse diese Nachbarn abstimmen.
- Vergleiche drei Arten dieser Abstimmung.
- Waehle die Methode mit dem besten `macro_f1`.

