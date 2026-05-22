import os
import pandas as pd
from tqdm import tqdm
from river import forest, naive_bayes, linear_model, preprocessing, metrics, optim, drift
from data_loader import load_sentiment140
from preprocessing import preprocess_tweet
from embeddings import load_glove_model, tweet_to_vector, vector_to_features
from config import ARF_METRICS_PATH, NB_METRICS_PATH, LR_METRICS_PATH
#d#
from config import ARF_DRIFT_PATH, NB_DRIFT_PATH, LR_DRIFT_PATH
from config import FINAL_ONLINE_METRICS_PATH


def create_metrics():
    """
    Inicializuje a vráti slovník kumulatívnych metrík pre hodnotenie modelu.
    """
    return {
        "accuracy": metrics.Accuracy(),
        "precision": metrics.Precision(pos_val=1),
        "recall": metrics.Recall(pos_val=1),
        "f1": metrics.F1(pos_val=1),
        "balanced_accuracy": metrics.BalancedAccuracy(),
    }


def update_metrics(metric_dict, y_true, y_pred):
    """
    Aktualizuje všetky metriky v slovníku na základe predikcie a skutočnej triedy.
    """
    for metric in metric_dict.values():
        metric.update(y_true, y_pred)


def get_metric_values(metric_dict):
    """
    Vráti aktuálne hodnoty všetkých metrík ako slovník.
    """
    return {
        name: metric.get()
        for name, metric in metric_dict.items()
    }


def save_metric_row(save_path, row_result):
    """
    Pripojí jeden riadok metrík do CSV súboru.
    Ak súbor neexistuje, vytvorí ho aj s hlavičkou.
    """
    pd.DataFrame([row_result]).to_csv(
        save_path,
        mode="a",
        header=not os.path.exists(save_path),
        index=False
    )


def run_experiment(
    model,
    model_name,
    save_path,
    #d#
    drift_save_path=None,
    nrows=None,
    skip_rows=0,
    log_every=100_000,
    save_every=500
):
    """
    Spustí online experiment pre jeden model na dátovom prúde.

    Parametre:
        model: Online model z knižnice River.
        model_name (str): Názov modelu (používa sa pri logovaní a ukladaní).
        save_path (str): Cesta k CSV súboru pre priebežné kumulatívne metriky.
        drift_save_path (str | None): Cesta k CSV súboru pre záznamy detekcie driftu.
        nrows (int | None): Počet načítaných riadkov datasetu. None = celý dataset.
        skip_rows (int): Počet riadkov preskočených na začiatku (vyhradených pre statické modely).
        log_every (int): Interval výpisu metrík do konzoly (počet spracovaných vzoriek).
        save_every (int): Interval ukladania metrík do CSV (počet spracovaných vzoriek).
    """
    df = load_sentiment140(nrows=nrows)

    # Preskočenie úvodných riadkov vyhradených pre tréning statických modelov
    if skip_rows > 0:
        df = df.iloc[skip_rows:].reset_index(drop=True)
        print(f"Skipped first {skip_rows} rows. Online training starts from row {skip_rows + 1}.")

    print(f"\nLoading GloVe embeddings for {model_name}...")
    glove_model = load_glove_model()
    print("GloVe loaded.\n")

    # Inicializácia kumulatívnych metrík sledovaných počas celého behu
    cumulative_metrics = create_metrics()
    #d#

    # Inicializácia detektora konceptového driftu ADWIN
    drift_detector = drift.ADWIN(delta=0.0002)

    # Vymazanie existujúcich výstupných súborov pred novým behom
    if drift_save_path is not None and os.path.exists(drift_save_path):
        os.remove(drift_save_path)

    if os.path.exists(save_path):
        os.remove(save_path)

    #d#
    
    #d#

    # Hlavná slučka - iterácia cez dátový prúd vzorka po vzorke (prequential evaluation)
    for i, row in enumerate(
        tqdm(df.itertuples(index=False), total=len(df), desc=f"Training {model_name}"),
        start=1
    ):
        raw_text = row.text
        y = int(row.target)
        date = row.date
        #d#
        
        #d#
        
        # Predspracovanie textu a vytvorenie GloVe vektora
        processed_text = preprocess_tweet(raw_text)
        vector = tweet_to_vector(processed_text, glove_model)

        # Prevod vektora na slovník príznakov vyžadovaný knižnicou River
        x = vector_to_features(vector)

        # Test: Predikcia modelu na aktuálnej vzorke (pred doučením)
        y_pred = model.predict_one(x)

        if y_pred is not None:
            # Aktualizácia kumulatívnych metrík na základe predikcie
            update_metrics(cumulative_metrics, y, y_pred)
            #d#
            #d#

            # Výpočet binárnej chyby a aktualizácia detektora driftu ADWIN
            error = int(y_pred != y)
            drift_detector.update(error)

            # Ak bol detegovaný drift, zaznamená sa do CSV súboru
            if drift_save_path is not None and drift_detector.drift_detected:
                drift_result = {
                    "model": model_name,
                    "tweet_index": i,
                    "original_tweet_index": i + skip_rows,
                    "date": date,
                    "error": error,
                    "adwin_estimation": drift_detector.estimation,
                    "adwin_width": drift_detector.width,
                    "n_detections": drift_detector.n_detections,
                }

                save_metric_row(drift_save_path, drift_result)

                print(
                    f"\nDrift detected | "
                    f"Model: {model_name} | "
                    f"Tweet index: {i} | "
                    f"Original index: {i + skip_rows} | "
                    f"Date: {date}"
                )

        # Train: doučenie modelu na aktuálnej vzorke
        model.learn_one(x, y)

        # Uloženie kumulatívnych metrík každých save_every vzoriek
        if i % save_every == 0:
            current_metrics = get_metric_values(cumulative_metrics)

            row_result = {
                "model": model_name,
                "tweet_index": i,
                "original_tweet_index": i + skip_rows,
                "date": date,
                "window_type": "cumulative_save_every",
                **current_metrics
            }

            save_metric_row(save_path, row_result)

        # Výpis aktuálnych metrík do konzoly každých log_every vzoriek
        if i % log_every == 0:
            print(
                f"Processed: {i} tweets | "
                f"Original index: {i + skip_rows} | "
                f"Model: {model_name} | "
                f"Accuracy: {cumulative_metrics['accuracy'].get():.4f} | "
                f"Precision: {cumulative_metrics['precision'].get():.4f} | "
                f"Recall: {cumulative_metrics['recall'].get():.4f} | "
                f"F1: {cumulative_metrics['f1'].get():.4f}"
            )


    #d#

    # Zobrazenie posledných uložených riadkov metrík po dokončení behu
    if os.path.exists(save_path):
        results_df = pd.read_csv(save_path)
        print(results_df.tail())
    else:
        results_df = pd.DataFrame()
        print("No metrics were saved. Try lowering save_every or increasing nrows.")

    print(f"\nTraining finished for {model_name}.")
    print(f"Metrics saved to: {save_path}")

    #d#

    # Uloženie finálnych kumulatívnych metrík po prechode celým dátovým prúdom
    final_result = {
        "model": model_name,
        "window_type": "final_cumulative",
        "tweets_processed": len(df),
        "skip_rows": skip_rows,
        **get_metric_values(cumulative_metrics)
    }

    save_metric_row(FINAL_ONLINE_METRICS_PATH, final_result)

    print("\nFinal cumulative metrics:")
    print(pd.DataFrame([final_result]))

    return results_df


def run_all_experiments(nrows=None, skip_rows=0):
    """
    Spustí experimenty pre všetky tri online modely sekvenčne.

    Parametre:
        nrows (int / None): Počet načítaných riadkov datasetu. None = celý dataset.
        skip_rows (int): Počet riadkov preskočených na začiatku pre každý model.
    """
    # Vymazanie súboru s finálnymi metrikami pred novým behom
    if os.path.exists(FINAL_ONLINE_METRICS_PATH):
        os.remove(FINAL_ONLINE_METRICS_PATH)

    # Definícia experimentov — každý obsahuje model, jeho názov a cesty k výstupným súborom
    experiments = [
        {
            "model_name": "naive_bayes",
            "model": naive_bayes.GaussianNB(),
            "save_path": NB_METRICS_PATH,
            #d#
            "drift_save_path": NB_DRIFT_PATH,
        },
        {
            "model_name": "logistic_regression",
            "model": preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=optim.SGD(.1)),
            "save_path": LR_METRICS_PATH,
            #d#
            "drift_save_path": LR_DRIFT_PATH,      
        },
        {
            "model_name": "arf",
            "model": forest.ARFClassifier(seed=42),
            "save_path": ARF_METRICS_PATH,
            #d#
            "drift_save_path": ARF_DRIFT_PATH,
        },
    ]

    for experiment in experiments:
        run_experiment(
            model=experiment["model"],
            model_name=experiment["model_name"],
            save_path=experiment["save_path"],
            #d#
            drift_save_path=experiment["drift_save_path"],
            nrows=nrows,
            skip_rows=skip_rows,
            log_every=100_000,
            save_every=500
        )


if __name__ == "__main__":
    run_all_experiments(
        #nrows=100000,
        # Prvých 5000 vzoriek vyhradených pre tréning statických modelov
        skip_rows=5_000
    )