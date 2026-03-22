# 🫀 ECG Heartbeat Classification System

> Système professionnel d'analyse et de classification des battements cardiaques MIT-BIH — Machine Learning meets Cardiologie.

---

## 📋 Table des matières

- [Aperçu](#-aperçu)
- [Architecture du projet](#-architecture-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Classes de battements](#-classes-de-battements)
- [Modèles disponibles](#-modèles-disponibles)
- [Pipeline de traitement](#-pipeline-de-traitement)
- [Résultats](#-résultats)
- [Configuration](#-configuration)

---

## 🔬 Aperçu

Ce système prend en entrée des enregistrements ECG bruts de la base **MIT-BIH Arrhythmia Database** et produit en sortie une classification automatique des battements cardiaques en 5 types, comparant plusieurs approches ML et Deep Learning.

Le pipeline est entièrement automatisé : téléchargement → nettoyage → extraction → entraînement → rapport.

**Ce que ça fait concrètement :**

- Télécharge et lit les enregistrements MIT-BIH via `wfdb`
- Nettoie le signal ECG (NeuroKit2 ou fallback brut)
- Détecte les R-peaks avec 3 méthodes en cascade (XQRS → GQRS → fallback)
- Extrait des segments de 300 points centrés sur chaque battement
- Entraîne 4 modèles ML classiques + 6 architectures Deep Learning
- Génère un rapport complet avec matrices de confusion, courbes ROC et comparaison

---

## 🗂️ Architecture du projet

```
ecg-classification/
│
├── ecg_pipeline.py       ← script principal
├── ecg_analysis.log      ← logs auto-générés
│
├── mitdb/                ← enregistrements MIT-BIH
├── cache/                ← cache disque (hash MD5)
├── models/               ← modèles sauvegardés
├── plots/                ← visualisations
└── report/               ← rapport final
    ├── performance_summary.csv
    ├── cm_*.png
    ├── roc_curves.png
    └── model_comparison.png
```

Le code est organisé en composants bien séparés :

```
ECGConfig         →  tous les paramètres au même endroit
CacheManager      →  cache mémoire + disque, évite les recalculs
ECGDataLoader     →  téléchargement, lecture, nettoyage du signal
FeatureExtractor  →  segments + features morphologiques & temporelles
SklearnModel      →  wrapper RF / XGBoost / Logistic / SVM
CNNModel          →  CNN 1D TensorFlow/Keras
ECGPipeline       →  orchestrateur principal
```

---

## ⚙️ Installation

```bash
# Cloner le dépôt
git clone https://github.com/Aziz-Marnissi/ecg-classification.git
cd ecg-classification

# Environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Dépendances minimales
pip install numpy pandas scipy scikit-learn matplotlib seaborn wfdb

# Dépendances complètes (recommandé)
pip install tensorflow xgboost neurokit2 librosa joblib
```

> Les dépendances optionnelles (TensorFlow, XGBoost, NeuroKit2) sont détectées automatiquement. Si elles sont absentes, le système bascule sur des méthodes de fallback sans planter.

---

## 🚀 Utilisation

### Lancement rapide

```bash
python ecg_pipeline.py
```

### Via l'API Python

```python
from ecg_pipeline import ECGConfig, ECGPipeline

config = ECGConfig(max_epochs=100, batch_size=64, cache_enabled=True)
pipeline = ECGPipeline(config)

# Charger et traiter les données
pipeline.load_data(["100", "101", "102"], download=True)
pipeline.extract_features()

# Entraîner
ml_data = pipeline.prepare_ml_data()
pipeline.train_classical_models(*ml_data)

dl_data = pipeline.prepare_dl_data()
pipeline.train_cnn(*dl_data)

# Générer le rapport
pipeline.generate_report(output_dir="report")
```

### Entraîner un seul modèle

```python
from ecg_pipeline import SklearnModel, ECGConfig

model = SklearnModel(model_type="rf", config=ECGConfig())
model.train(X_train, y_train, class_weights=weights)

results = model.evaluate(X_test, y_test, label_encoder)
print(f"F1  :  {results['f1_score']:.3f}")
print(f"Acc :  {results['accuracy']:.3f}")

model.save("models/rf.joblib")
```

Les types de modèles disponibles sont `'rf'`, `'xgboost'`, `'logistic'` et `'svm'`.

---

## 🏷️ Classes de battements

5 types de battements selon la nomenclature MIT-BIH :

| Label | Nom complet | Description |
|:---:|---|---|
| `N` | Normal | Battement sinusal standard, pic R net, ligne de base stable |
| `V` | Ventricular Ectopic | Contraction ventriculaire prématurée, morphologie inversée |
| `A` | Atrial Ectopic | Contraction auriculaire prématurée, intervalle RR raccourci |
| `L` | Left Bundle Branch Block | Bloc de branche gauche, QRS élargi |
| `R` | Right Bundle Branch Block | Bloc de branche droite, aspect rSR' caractéristique |

Le dataset est naturellement très déséquilibré (N est de loin la classe majoritaire). Le système gère ça automatiquement via `class_weight='balanced'` pour les modèles sklearn, et `sample_weight` pour XGBoost.

---

## 🤖 Modèles disponibles

### Machine Learning classique

| Modèle | Config notable |
|---|---|
| **Random Forest** | 200 arbres, profondeur max 15, `class_weight='balanced'` |
| **XGBoost** | 300 estimateurs, lr=0.05, subsample=0.9 |
| **Logistic Regression** | multinomial, 1000 itérations max |
| **SVM** | kernel RBF, probabilités activées |

### Deep Learning — CNN 1D

L'architecture de base est un CNN 1D à 3 blocs convolutifs :

```
Input (300, 1)
  ↓
Conv1D(64, kernel=7)  →  BatchNorm  →  MaxPool  →  Dropout(0.3)
  ↓
Conv1D(128, kernel=5) →  BatchNorm  →  MaxPool  →  Dropout(0.3)
  ↓
Conv1D(256, kernel=3) →  BatchNorm  →  MaxPool  →  Dropout(0.4)
  ↓
GlobalAveragePooling
  ↓
Dense(256, relu, L2)  →  Dropout(0.5)
  ↓
Dense(n_classes, softmax)
```

Optimizer Adam (lr=0.001), perte Sparse Categorical Crossentropy.

Deux callbacks guident l'entraînement : **EarlyStopping** (patience=15, restaure les meilleurs poids) et **ReduceLROnPlateau** (patience=5, divise le lr par 2).

### Variantes expérimentées

| Modèle | Spécificité |
|---|---|
| **CNN++** | Architecture CNN plus profonde |
| **CNN 2D** | Convolutions 2D sur représentation temps-fréquence |
| **CNN Amélioré** | Connexions résiduelles |
| **CNN+LSTM** | CNN pour l'extraction locale, LSTM pour les dépendances temporelles |
| **Hybride** | Fusion features morphologiques + segments bruts |

---

## 🔄 Pipeline de traitement

```
Enregistrements MIT-BIH
        │
        ▼
  Téléchargement wfdb  (retry x3 automatique)
        │
        ▼
  Nettoyage signal  (NeuroKit2 → fallback brut)
        │
        ▼
  Détection R-peaks  (XQRS → GQRS → find_peaks)
        │
        ▼
  Association annotations  (tolérance 100ms)
        │
        ▼
  Extraction segments  (200ms avant + 400ms après le R-peak)
        │
        ▼
  Rééchantillonnage  →  300 points par segment
        │
        ├───────────────────────────┐
        ▼                           ▼
  Features morpho               Segments bruts
  (stats, FFT, skewness)        (pour CNN)
        │                           │
        ▼                           ▼
  Train / Val / Test split    Train / Val / Test split
        │                           │
        ▼                           ▼
  StandardScaler              Reshape  →  (N, 300, 1)
        │                           │
        ▼                           ▼
  RF / XGBoost / LR / SVM     CNN / CNN++ / CNN+LSTM...
        │                           │
        └─────────────┬─────────────┘
                      ▼
               Rapport de performance
```

---

## 📊 Résultats

### Ce qu'on observe sur les matrices de confusion

Tous les modèles souffrent du même problème : le fort déséquilibre de classes (448 N contre seulement 7 A dans le jeu de test) pousse les modèles à favoriser la classe majoritaire.

**Deep Learning — bilan par modèle :**

| Modèle | Rappel sur A | Rappel sur N | Verdict |
|---|:---:|:---:|---|
| CNN baseline | 100% ✅ | 0% ❌ | biais total vers A |
| CNN++ | 71% ✅ | 99.8% ✅ | meilleur équilibre |
| CNN 2D | 0% ❌ | 100% ✅ | ignore totalement A |
| CNN Amélioré | 0% ❌ | 100% ✅ | ignore totalement A |
| CNN+LSTM | 14% ⚠️ | 84% ⚠️ | plus de faux positifs |
| Hybride | 100% ✅ | 99.8% ✅ | proche du CNN++ |

**ML classique :** Random Forest, Logistic Regression, SVM et XGBoost donnent tous le même résultat : 7/7 sur A et 447/448 sur N — performances solides et stables.

### Courbes d'apprentissage CNN

La training loss descend rapidement mais la validation loss remonte dès la 2e epoch — signe clair d'overfitting. La validation accuracy reste proche de 0 : le modèle prédit quasi-exclusivement N sur le jeu de validation.

### Courbe ROC

AUC = 0.42 sur les deux classes, ce qui est en dessous d'un classifieur aléatoire. Ce chiffre confirme que le modèle n'a pas appris de représentation utile pour séparer les classes, faute de données suffisantes sur les classes minoritaires.

### Pistes d'amélioration

- **SMOTE / ADASYN** — rééchantillonner les classes minoritaires avant l'entraînement
- **Focal Loss** — pénaliser davantage les erreurs sur les classes rares
- **Plus de données** — utiliser les 48 enregistrements complets (10 chargés par défaut)
- **Data augmentation** — bruit gaussien et time-stretching sur les segments
- **Ensemble** — combiner CNN++ et Random Forest par stacking

---

## 🛠️ Configuration

Tout est centralisé dans `ECGConfig`, aucun paramètre éparpillé dans le code :

```python
@dataclass
class ECGConfig:
    # Signal
    fs: int = 360                     # fréquence d'échantillonnage (Hz)
    window_before_ms: float = 200.0   # fenêtre avant le R-peak
    window_after_ms: float = 400.0    # fenêtre après le R-peak
    desired_length: int = 300         # longueur cible après rééchantillonnage

    # Classes
    valid_labels: List[str] = ["N", "V", "A", "L", "R"]

    # Splits
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42

    # Deep Learning
    max_epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    min_lr: float = 1e-7

    # Système
    cache_enabled: bool = True
    n_workers: int = 4
```

---

## 📄 Licence

MIT — Données MIT-BIH disponibles sur [PhysioNet](https://physionet.org/content/mitdb/1.0.0/).

---

> Pour de meilleures performances : charger les 48 enregistrements complets, utiliser un GPU pour TensorFlow, et corriger le déséquilibre de classes avec SMOTE avant d'entraîner les modèles Deep Learning.
