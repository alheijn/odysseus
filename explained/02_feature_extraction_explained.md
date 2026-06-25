# Erklaerung: `02_feature_extraction.ipynb`

## Worum geht es in diesem Notebook?

Dieses Notebook berechnet **Merkmale (Features)** aus Bildern griechischer Buchstaben.

Die Grundidee ist:

1. Bilder werden geladen.
2. Ein vortrainiertes CNN-Modell (`VGG16`) schaut sich die Bilder an.
3. Statt direkt eine Klassifikation zu machen, wird aus jedem Bild ein Zahlenvektor erzeugt.
4. Diese Zahlenvektoren werden gespeichert und spaeter von klassischen Machine-Learning-Modellen wie k-NN verwendet.

Das Notebook macht das **dreimal**, mit drei Varianten des Feature-Extractors:

- `VGG16FeatureExtractor`: spaete, tiefere Merkmale
- `EarlyVGG16FeatureExtractor`: fruehere, einfachere Merkmale
- `VGG16SPPFeatureExtractor`: tiefere Merkmale mit mehr raeumlicher Information

---

## Vorwissen: Was ist "Feature Extraction"?

Ein Bild besteht zuerst nur aus Pixeln.
Ein Modell wie VGG16 wandelt diese Pixel Schritt fuer Schritt in sinnvollere Informationen um, zum Beispiel:

- Kanten
- Striche
- Formen
- komplexere Muster

Das Ergebnis ist dann ein **Feature-Vektor**, also z. B. eine Liste aus 256, 512 oder 2048 Zahlen.
Diese Zahlen sollen das Bild so beschreiben, dass ein spaeteres Modell damit besser arbeiten kann als mit rohen Pixeln.

---

## Zelle 0: Imports und Projektstruktur

```python
%load_ext autoreload
%autoreload 2

import sys
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
```

### Was passiert hier?

- `%load_ext autoreload` und `%autoreload 2` sind Jupyter-Hilfen.
  Sie sorgen dafuer, dass geaenderte Python-Dateien automatisch neu geladen werden.
- `torch` wird fuer das neuronale Netz benutzt.
- `numpy` wird fuer Arrays und das Speichern der Features benutzt.
- `DataLoader` verarbeitet die Bilder in kleinen Paketen, also **Batches**.
- `tqdm` zeigt einen Fortschrittsbalken.

Danach:

```python
PROJECT_ROOT = os.path.abspath('..')
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
```

Damit wird der Projektordner zum Python-Pfad hinzugefuegt.
So kann das Notebook auf Code aus `src/` zugreifen.

Dann:

```python
from src.data_pipeline import CleanedGreekLettersDataset, VGG16FeatureExtractor
```

Hier werden zwei wichtige Bausteine importiert:

- `CleanedGreekLettersDataset`: laedt die Bilder und Labels
- `VGG16FeatureExtractor`: erzeugt Features mit VGG16

---

## Der Dataset-Code aus `src/data_pipeline.py`

Damit das Notebook verstaendlich ist, muss man kurz wissen, was dieses Dataset macht.

### `CleanedGreekLettersDataset`

Diese Klasse:

- durchsucht die Bildordner der 24 griechischen Buchstaben
- merkt sich zu jedem Bild den Dateipfad
- ordnet jedem Buchstaben eine Zahl zu, z. B. `Alpha -> 0`, `Beta -> 1`, ...
- ignoriert Unterordner mit `_flagged_` oder `_review_`

Beim Abrufen eines Eintrags:

```python
image = Image.open(img_path).convert("RGB")
label = self.labels[idx]
```

- das Bild wird geladen
- in RGB umgewandelt
- mit seinem Label zurueckgegeben

Wenn ein `transform` angegeben wurde, wird das Bild vorher noch skaliert und normalisiert.

---

## Zelle 1: Standard-Feature-Extraktion mit VGG16

Das ist der erste eigentliche Durchlauf.

### 1. Geraet waehlen

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

Hier wird geprueft:

- wenn Apple Metal (`mps`) verfuegbar ist, nutze GPU
- sonst CPU

Das ist nur fuer die Geschwindigkeit wichtig, nicht fuer die Logik.

---

### 2. Bildvorverarbeitung

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

Hier wird festgelegt, wie jedes Bild vorbereitet wird:

- `Resize((224, 224))`:
  VGG16 erwartet Bilder in dieser Groesse.
- `ToTensor()`:
  macht aus dem Bild einen PyTorch-Tensor.
- `Normalize(...)`:
  skaliert die Werte passend zu dem Format, mit dem VGG16 trainiert wurde.

Das ist wichtig, weil das Modell mit diesen Normalisierungswerten aus ImageNet trainiert wurde.

---

### 3. Datenordner und Ausgabeordner

```python
DATA_DIR = "../data/ALPUB_v2/images"
OUTPUT_DIR = "../data/extracted_features"
```

- `DATA_DIR`: hier liegen die Bilder
- `OUTPUT_DIR`: hier werden die berechneten Features gespeichert

---

### 4. Dataset und DataLoader

```python
dataset = CleanedGreekLettersDataset(root_dir=DATA_DIR, transform=transform)
assert len(dataset) > 0, ...
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
```

#### Bedeutung:

- `dataset` kennt alle Bilder und Labels
- `assert len(dataset) > 0` bricht ab, wenn keine Bilder gefunden wurden
- `DataLoader` liefert die Daten batchweise

Wichtige Parameter:

- `batch_size=128`:
  immer 128 Bilder auf einmal
- `shuffle=False`:
  Reihenfolge bleibt stabil
- `num_workers=4`:
  vier Prozesse helfen beim Laden

Warum `shuffle=False`?
Weil hier nichts trainiert wird. Es sollen nur Features erzeugt und in derselben Reihenfolge wie die Labels gespeichert werden.

---

### 5. Modell initialisieren

```python
model = VGG16FeatureExtractor().to(device)
model.eval()
```

`VGG16FeatureExtractor` aus `src/data_pipeline.py` macht intern:

```python
vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
self.features = vgg.features
self.pool = nn.AdaptiveAvgPool2d((1, 1))
```

### Was bedeutet das?

- Es wird ein **vortrainiertes VGG16-Modell** geladen.
- Verwendet wird nur der **Feature-Teil** des Netzes, nicht der Klassifikationskopf.
- Danach kommt `AdaptiveAvgPool2d((1, 1))`.

Dieses Pooling mittelt jede Feature-Map auf genau einen Wert herunter.

Wenn VGG16 am Ende 512 Kanaele hat, entsteht dadurch:

- vorher: viele Karten mit raeumlicher Struktur
- nachher: genau **512 Zahlen**

Im `forward`:

```python
x = self.features(x)
x = self.pool(x)
return torch.flatten(x, 1)
```

Also:

1. Bild durch VGG16
2. Pooling
3. alles zu einem flachen Vektor machen

Ergebnis pro Bild: ein Feature-Vektor mit **512 Dimensionen**

`model.eval()` sagt: Das Modell ist im Auswertungsmodus, nicht im Trainingsmodus.

---

### 6. Feature-Extraktionsschleife

```python
all_features = []
all_labels = []
```

Hier werden die Ergebnisse gesammelt.

Dann:

```python
with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="Extracting VGG-16 Features"):
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
```

### Was passiert pro Batch?

1. `images, labels` kommen aus dem DataLoader
2. `images = images.to(device)` schiebt die Bilder auf GPU oder CPU
3. `features = model(images)` berechnet die Merkmale
4. `features.cpu().numpy()` wandelt das Ergebnis in NumPy um
5. Features und Labels werden in Listen gespeichert

### Warum `torch.no_grad()`?

Weil hier **nicht trainiert** wird.
Es werden keine Gradienten gebraucht.
Das spart Speicher und Rechenzeit.

---

### 7. Zusammenbauen und speichern

```python
X = np.vstack(all_features)
y = np.concatenate(all_labels)
```

Aus vielen kleinen Batch-Ergebnissen wird ein grosses Ganzes:

- `X`: alle Feature-Vektoren untereinander
- `y`: alle Labels hintereinander

Dann:

```python
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "X_features_vgg16.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)
```

Das speichert:

- `X_features_vgg16.npy`: die Features
- `y_labels.npy`: die Labels

Diese Dateien koennen spaeter in anderen Notebooks direkt geladen werden.

---

## Zelle 2: `EarlyVGG16FeatureExtractor`

Jetzt wird fast derselbe Ablauf noch einmal gemacht, aber mit einem anderen Modell:

```python
from src.data_pipeline import EarlyVGG16FeatureExtractor
```

Im Modell steht:

```python
self.features = vgg.features[:17]
self.pool = nn.AdaptiveAvgPool2d((1, 1))
```

### Unterschied zum ersten Durchlauf

Hier werden **nur die ersten 3 VGG-Blöcke** benutzt.

Das bedeutet:

- das Modell schaut weniger tief ins Netz
- die Merkmale sind eher einfach:
  - Kanten
  - Striche
  - lokale Formen

Am Ende entstehen pro Bild **256 Features** statt 512.

### Warum macht man das?

Weil fruehe Features manchmal fuer einfache Formen besser geeignet sind als sehr abstrakte spaete Features.
Gerade bei Symbolen oder Buchstaben kann das sinnvoll sein.

### Sonst bleibt fast alles gleich

- gleiche Daten
- gleiches Laden
- gleiches Batch-Verfahren
- gleiches Speichern

Nur der Ausgabeordner ist anders:

```python
OUTPUT_DIR = "../data/extracted_features_early"
```

---

## Zelle 3: `VGG16SPPFeatureExtractor`

Die dritte Variante benutzt:

```python
from src.data_pipeline import VGG16SPPFeatureExtractor
```

Dieses Modell hat:

```python
self.features = vgg.features
self.pool = nn.AdaptiveAvgPool2d((2, 2))
```

### Der entscheidende Unterschied

Im ersten Modell wurde auf `1x1` gepoolt.
Hier wird auf `2x2` gepoolt.

Das bedeutet:

- pro Kanal bleibt nicht nur **ein** Mittelwert
- sondern **vier** Werte in einem 2x2-Raster

Wenn VGG16 am Ende 512 Kanaele hat, ergibt das:

- `512 * 2 * 2 = 2048` Werte

### Warum ist das interessant?

Damit bleibt mehr **raeumliche Information** erhalten.
Das Modell merkt sich also nicht nur, **welche** Merkmale da sind, sondern etwas besser auch, **wo** sie im Bild liegen.

Das kann bei Symbolen hilfreich sein, wenn die Position von Strichen oder Formen wichtig ist.

### Weitere kleine Aenderung

```python
transforms.Resize((256,256))
```

Hier wird auf `256x256` vergroessert.
Das passt gut zur Idee, etwas mehr raeumliche Struktur zu behalten.

Gespeichert wird in:

```python
OUTPUT_DIR = "../data/extracted_features_SPP"
```

---

## Zelle 4

Die letzte Zelle ist leer und macht nichts.

---

## Was ist das Ergebnis des gesamten Notebooks?

Am Ende existieren drei verschiedene Feature-Sammlungen:

1. **Standard VGG16 Features**
   - tiefe Merkmale
   - 512 Dimensionen

2. **Early VGG16 Features**
   - fruehere, einfachere Merkmale
   - 256 Dimensionen

3. **SPP Features**
   - tiefe Merkmale mit mehr Positionsinformation
   - 2048 Dimensionen

Diese Features sind die Grundlage fuer spaetere Notebooks wie k-NN.

---

## Warum ist dieses Notebook wichtig fuer den Rest des Projekts?

Ohne dieses Notebook haette man nur Bilder.
Mit diesem Notebook bekommt man numerische Beschreibungen der Bilder.

Klassische Modelle wie:

- k-NN
- SVM
- Logistic Regression

koennen nicht direkt gut mit Rohbildern arbeiten.
Sie brauchen Zahlenvektoren.
Genau diese Zahlenvektoren erzeugt dieses Notebook.

---

## Die Hauptidee in einem Satz

`02_feature_extraction.ipynb` nimmt Bilder griechischer Buchstaben, laesst VGG16 daraus kompakte Merkmalsvektoren berechnen und speichert diese Vektoren fuer die spaetere Klassifikation.

---

## Kurzfassung fuer Anfaenger

- Ein Bild wird geladen.
- Das Bild wird fuer VGG16 vorbereitet.
- VGG16 erzeugt daraus viele sinnvolle Zahlen.
- Diese Zahlen beschreiben das Bild.
- Die Zahlen werden zusammen mit den Labels gespeichert.
- Dasselbe wird mit drei verschiedenen Feature-Strategien ausprobiert.

