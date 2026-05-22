import sys
from pathlib import Path


# Koreňový adresár projektu (o úroveň vyššie ako tento súbor)
BASE_DIR = Path(__file__).resolve().parent.parent
# Pridanie koreňového adresára do systémovej cesty pre správne importy modulov
sys.path.append(str(BASE_DIR))

# Adresáre pre dáta a výsledky
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Cesta k surovému datasetu Sentiment140
RAW_DATA_PATH  = DATA_DIR / "sentiment140.csv"

# Cesty k súborom s priebežnými kumulatívnymi metrikami online modelov
ARF_METRICS_PATH = RESULTS_DIR / "arf_metrics.csv"
NB_METRICS_PATH = RESULTS_DIR / "naive_bayes_metrics.csv"
LR_METRICS_PATH = RESULTS_DIR / "logistic_regression_metrics.csv"

# Cesty k súborom pre detekciu konceptového driftu
ARF_DRIFT_PATH = RESULTS_DIR / "arf_drift_points.csv"
NB_DRIFT_PATH = RESULTS_DIR / "naive_bayes_drift_points.csv"
LR_DRIFT_PATH = RESULTS_DIR / "logistic_regression_drift_points.csv"

# Dostupné varianty predtrénovaných GloVe Twitter embeddingov
GLOVE_50_PATH  = DATA_DIR / "glove.twitter.27B.50d.txt"
GLOVE_100_PATH = DATA_DIR / "glove.twitter.27B.100d.txt"

#Aktívny embedding
GLOVE_PATH = GLOVE_50_PATH

# Názvy stĺpcov datasetu Sentiment140 (súbor neobsahuje hlavičku)
COLUMNS = ["target", "ids", "date", "flag", "user", "text"]

# Cesta k súboru s finálnymi kumulatívnymi metrikami online modelov po dokončení trénovania
FINAL_ONLINE_METRICS_PATH = "../results/final_online_metrics.csv"