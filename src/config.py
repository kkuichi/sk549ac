import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RAW_DATA_PATH  = DATA_DIR / "sentiment140.csv"
ARF_METRICS_PATH = RESULTS_DIR / "arf_metrics.csv"
NB_METRICS_PATH = RESULTS_DIR / "naive_bayes_metrics.csv"
LR_METRICS_PATH = RESULTS_DIR / "logistic_regression_metrics.csv"

ARF_DAILY_METRICS_PATH = RESULTS_DIR / "arf_daily_metrics.csv"
NB_DAILY_METRICS_PATH = RESULTS_DIR / "naive_bayes_daily_metrics.csv"
LR_DAILY_METRICS_PATH = RESULTS_DIR / "logistic_regression_daily_metrics.csv"

ARF_DRIFT_PATH = RESULTS_DIR / "arf_drift_points.csv"
NB_DRIFT_PATH = RESULTS_DIR / "naive_bayes_drift_points.csv"
LR_DRIFT_PATH = RESULTS_DIR / "logistic_regression_drift_points.csv"

GLOVE_50_PATH  = DATA_DIR / "glove.twitter.27B.50d.txt"
GLOVE_100_PATH = DATA_DIR / "glove.twitter.27B.100d.txt"
#Aktívny embedding
GLOVE_PATH = GLOVE_50_PATH

COLUMNS = ["target", "ids", "date", "flag", "user", "text"]

FINAL_ONLINE_METRICS_PATH = "../results/final_online_metrics.csv"
