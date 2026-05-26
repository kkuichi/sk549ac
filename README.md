# Použivateľská a systémová príručka

Systémová príručka na popis a spustenie všetkých kódov v tomto úložisku.

Repozitár obsahuje implementáciu systému pre analýzu sentimentu na dátových prúdoch krátkych textov. Súčasťou je simulácia chronologického dátového prúdu na datasete Sentiment140, predspracovanie textu, vektorizácia pomocou predtrénovaných GloVe Twitter embeddingov a trénovanie troch online modelov (Gaussian Naive Bayes, Logistická regresia, Adaptive Random Forest) s priebežnou detekciou konceptového driftu pomocou metódy ADWIN. Online modely sú následne porovnané s ich statickými ekvivalentmi natrénovanými pomocou knižnice scikit-learn.

Názov bakalárskej práce: 	Analýza sentimentu na dátových prúdoch krátkych textov.

---

## Požiadavky

### 1. Vytvorenie virtuálneho prostredia

Pred inštaláciou knižníc je odporúčané vytvoriť virtuálne prostredie pre Python 3.12:

```bash
py -3.12 -m venv venv
```

Následne virtuálne prostredie aktivujte v priečinku projektu:

```bash
venv\Scripts\Activate.ps1
```

### 2. Inštalácia knižníc

Po aktivácii virtuálneho prostredia nainštalujte potrebné knižnice:

```bash
pip install river
pip install numpy
pip install scikit-learn
pip install gensim
pip install matplotlib
```
Ak sa počas inštalácie river nenainštaluje pandas tak aj:
```bash
pip install pandas
```
## Štruktúra projektu

```
project/
├── src/
│   ├── config.py            # Konfigurácia ciest a konštánt
│   ├── data_loader.py       # Načítanie a príprava datasetu
│   ├── preprocessing.py     # Predspracovanie textu tweetov
│   ├── embeddings.py        # GloVe embeddingy a vektorizácia
│   ├── train_online.py      # Trénovanie online modelov
│   ├── train_static.py      # Trénovanie statických modelov
│   └── plot_metrics.py      # Vizualizácia metrík a detekcie driftu
├── data/
│   ├── sentiment140.csv             # Dataset Sentiment140
│   ├── glove.twitter.27B.50d.txt    # GloVe Twitter embeddingy (50d)
│   └── glove.twitter.27B.100d.txt   # GloVe Twitter embeddingy (100d) — voliteľné
└── results/                 # Priečinok pre výstupné CSV súbory a grafy (vytvorí sa automaticky)
```

---

## Príprava dát

### Dataset Sentiment140

Dataset Sentiment140 je dostupný na platforme Kaggle:
[https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

Stiahnutý súbor `training.1600000.processed.noemoticon.csv` premenujte na `sentiment140.csv` a umiestnite ho do priečinka `data/`.

### GloVe Twitter embeddingy

Predtrénované GloVe Twitter embeddingy sú dostupné na stránke Stanford NLP:
[https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

Stiahnite súbor `glove.twitter.27B.zip`, rozbaľte ho a súbor `glove.twitter.27B.50d.txt` umiestnite do priečinka `data/`. Aktívny variant embeddingov je možné zmeniť v súbore `config.py` (premenná `GLOVE_PATH`).

---

## Trénovanie online modelov (train_online.py)

Skript trénuje tri online modely na chronologicky zoradenom dátovom prúde pomocou princípu prequential evaluation (test-then-train). Prvých 5000 vzoriek je preskočených — tie sú vyhradené pre tréning statických modelov. Súbežne s trénovaním prebieha detekcia konceptového driftu pomocou metódy ADWIN.

### Použitie
v priečinku projektu:
```bash
cd src
python3 train_online.py
```

### Modely

- **Gaussian Naive Bayes** — pravdepodobnostný klasifikátor z knižnice River
- **Logistická regresia** — pipeline `StandardScaler → LogisticRegression` s optimalizérom SGD (lr=0.1)
- **Adaptive Random Forest** — ansámblový model navrhnutý pre dátové prúdy (seed=42)

### Výstupy

Metriky sú ukladané každých 500 spracovaných vzoriek:

- `results/arf_metrics.csv` — kumulatívne metriky ARF
- `results/naive_bayes_metrics.csv` — kumulatívne metriky Naive Bayes
- `results/logistic_regression_metrics.csv` — kumulatívne metriky logistickej regresie
- `results/arf_drift_points.csv` — body detekcie driftu pre ARF
- `results/naive_bayes_drift_points.csv` — body detekcie driftu pre Naive Bayes
- `results/logistic_regression_drift_points.csv` — body detekcie driftu pre logistickú regresiu
- `results/final_online_metrics.csv` — finálne kumulatívne metriky všetkých online modelov

---

## Trénovanie statických modelov (train_static.py)

Skript trénuje dva statické modely na úvodných 5000 vzorkách datasetu pomocou knižnice scikit-learn. Po natrénovaní sú modely vyhodnotené na zvyšku dátového prúdu, pričom metriky sú počítané kumulatívne v rovnakých intervaloch ako pri online modeloch.

### Použitie
v priečinku projektu:
```bash
cd src
python3 train_static.py
```

### Modely

- **Random Forest** — `RandomForestClassifier` so 100 stromami (random_state=42)
- **Logistická regresia** — pipeline `StandardScaler → LogisticRegression` (max_iter=1000, random_state=42)

### Výstupy

- `results/static_random_forest_metrics.csv` — kumulatívne metriky statického Random Forestu
- `results/static_logistic_regression_metrics.csv` — kumulatívne metriky statickej logistickej regresie
- `results/final_static_metrics.csv` — finálne kumulatívne metriky statických modelov

---

## Vizualizácia výsledkov (plot_metrics.py)

Skript načíta výstupné CSV súbory všetkých modelov a vygeneruje porovnávacie grafy kumulatívneho vývoja metrík v čase. Do grafov sú zakreslené konsenzuálne pásma konceptového driftu — oblasti, v ktorých všetky tri online modely detegovали drift v rozpätí 10 000 vzoriek.

### Použitie
v priečinku projektu:
```bash
cd src
python3 plot_metrics.py
```

### Výstupy

Grafy sú uložené do priečinka `results/`:

- `results/accuracy_comparison_with_drift_by_date.png`
- `results/precision_comparison_with_drift_by_date.png`
- `results/recall_comparison_with_drift_by_date.png`
- `results/f1_comparison_with_drift_by_date.png`
- `results/balanced_accuracy_comparison_with_drift_by_date.png`

---

## Odporúčané poradie spustenia

```bash
# 1. Vytvorenie a aktivácia virtuálneho prostredia
py -3.12 -m venv venv
venv\Scripts\Activate.ps1

# 2. Inštalácia závislostí
pip install river
pip install numpy
pip install scikit-learn
pip install gensim
pip install matplotlib

# 3. Príprava dát
# - Stiahnuť sentiment140.csv a umiestniť do data/
# - Stiahnuť glove.twitter.27B.50d.txt a umiestniť do data/

# 4. Trénovanie online modelov
cd src
python3 train_online.py

# 5. Trénovanie statických modelov
python3 train_static.py

# 6. Vizualizácia výsledkov
python3 plot_metrics.py
```

---

## Reprodukovateľnosť

Adaptive Random Forest používa pevný seed `42` (`seed=42`). Statické modely používajú `random_state=42`. Výsledky sa môžu mierne líšiť v závislosti od verzie knižníc a operačného systému.

## Odkaz na repozitár k bakalárskej práci
https://github.com/kkuichi/sk549ac
