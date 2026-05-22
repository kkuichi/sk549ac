import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data_loader import load_sentiment140
from preprocessing import preprocess_tweet
from embeddings import load_glove_model, tweet_to_vector


# Počet vzoriek vyhradených na tréning statických modelov
TRAIN_SIZE = 5_000

# Cesty k výstupným CSV súborom s metrikami statických modelov
STATIC_RF_METRICS_PATH = "../results/static_random_forest_metrics.csv"
STATIC_LR_METRICS_PATH = "../results/static_logistic_regression_metrics.csv"
FINAL_STATIC_METRICS_PATH = "../results/final_static_metrics.csv"

def texts_to_vectors(texts, glove_model, desc="Creating vectors"):
    """
    Prevedie zoznam textov na maticu GloVe vektorov.

    Parametre:
        texts: Séria alebo zoznam textov na vektorizáciu.
        glove_model (KeyedVectors): Načítaný model GloVe embeddingov.
        desc (str): Popis zobrazovaný v progress bare.

    Návratová hodnota:
        np.ndarray: Matica vektorov tvaru (počet_textov, dimenzia_vektora).
    """
    vectors = []

    for text in tqdm(texts, desc=desc):
        # Predspracovanie textu pred vektorizáciou
        processed_text = preprocess_tweet(text)
        vector = tweet_to_vector(processed_text, glove_model)
        vectors.append(vector)

    # Zlúčenie zoznamu vektorov do jednej matice
    return np.vstack(vectors)


def calculate_metrics(y_true, y_pred):
    """
    Vypočíta klasifikačné metriky pre dané predikcie.

    Parametre:
        y_true: Skutočné triedy.
        y_pred: Predikované triedy.

    Návratová hodnota:
        dict: Slovník s hodnotami metrík accuracy, precision, recall, f1, balanced_accuracy.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # Precision a recall sú počítané pre pozitívnu triedu (label=1)
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
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


def evaluate_static_model_over_time(
    model,
    model_name,
    X_test,
    y_test,
    test_dates,
    save_path,
    save_every=500
):
    """
    Vyhodnotí statický model kumulatívne v rovnakých intervaloch ako online modely.

    Predikcia sa vykoná jednorazovo pre celú testovaciu množinu a metriky sa
    následne počítajú na postupne každých X predikcií. 
    Výsledky sú priebežne ukladané do CSV súboru.

    Parametre:
        model: Natrénovaný statický model zo scikit-learn.
        model_name (str): Názov modelu.
        X_test (np.ndarray): Testovacia matica príznakov.
        y_test (np.ndarray): Skutočné triedy testovacej množiny.
        test_dates (pd.Series): Dátumy zodpovedajúce testovacím vzorkám.
        save_path (str): Cesta k výstupnému CSV súboru s metrikami.
        save_every (int): Interval ukladania metrík.
    """
    # Vymazanie existujúceho súboru pred novým behom
    if os.path.exists(save_path):
        os.remove(save_path)

    # Jednorazová predikcia pre celú testovaciu množinu
    print(f"Predicting all test samples for {model_name}...")
    y_pred_all = model.predict(X_test)
    print("Prediction finished.")

    # Kumulatívny výpočet metrík — predikcie sa postupne odkrývajú po save_every vzorkách
    for i in tqdm(
        range(save_every, len(X_test) + 1, save_every),
        desc=f"Calculating metrics for {model_name}"
    ):
        # Kumulatívne okno od začiatku po aktuálny bod
        y_true_window = y_test[:i]
        y_pred_window = y_pred_all[:i]

        current_metrics = calculate_metrics(y_true_window, y_pred_window)

        row_result = {
            "model": model_name,
            "tweet_index": i,
            # Skutočný index v pôvodnom datasete (posun o TRAIN_SIZE)
            "original_tweet_index": i + TRAIN_SIZE,
            "date": test_dates.iloc[i - 1],
            "window_type": "static_cumulative_save_every",
            **current_metrics
        }

        save_metric_row(save_path, row_result)

    # Spracovanie zvyšných vzoriek ak celkový počet nie je deliteľný save_every
    if len(X_test) % save_every != 0:
        i = len(X_test)

        current_metrics = calculate_metrics(y_test, y_pred_all)

        row_result = {
            "model": model_name,
            "tweet_index": i,
            "original_tweet_index": i + TRAIN_SIZE,
            "date": test_dates.iloc[i - 1],
            "window_type": "static_cumulative_save_every",
            **current_metrics
        }

        save_metric_row(save_path, row_result)

    print(f"Metrics saved to: {save_path}")

    # Uloženie finálnych metrík po prechode celou testovacou množinou
    final_metrics = calculate_metrics(y_test, y_pred_all)

    final_result = {
        "model": model_name,
        "window_type": "final_static_cumulative",
        "train_size": TRAIN_SIZE,
        "test_size": len(X_test),
        **final_metrics
    }

    save_metric_row(FINAL_STATIC_METRICS_PATH, final_result)

    print("\nFinal static cumulative metrics:")
    print(pd.DataFrame([final_result]))


def run_static_experiments(nrows=None, save_every=500):
    """
    Spustí experimenty pre všetky statické modely.

    Parametre:
        nrows (int / None): Počet načítaných riadkov datasetu. None = celý dataset.
        save_every (int): Interval ukladania metrík (počet vzoriek).
    """
    # Vymazanie súboru s finálnymi metrikami pred novým behom
    if os.path.exists(FINAL_STATIC_METRICS_PATH):
        os.remove(FINAL_STATIC_METRICS_PATH)

    df = load_sentiment140(nrows=nrows)

    # Rozdelenie datasetu: prvých TRAIN_SIZE vzoriek na tréning, zvyšok na testovanie
    train_df = df.iloc[:TRAIN_SIZE].reset_index(drop=True)
    test_df = df.iloc[TRAIN_SIZE:].reset_index(drop=True)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    print("\nLoading GloVe embeddings...")
    glove_model = load_glove_model()
    print("GloVe loaded.\n")

    # Vektorizácia trénovacej a testovacej množiny pomocou GloVe embeddingov
    X_train = texts_to_vectors(
        train_df["text"],
        glove_model,
        desc="Creating train vectors"
    )
    y_train = train_df["target"].astype(int).values

    X_test = texts_to_vectors(
        test_df["text"],
        glove_model,
        desc="Creating test vectors"
    )
    y_test = test_df["target"].astype(int).values
    test_dates = test_df["date"]

    # Definícia statických modelov a ich výstupných ciest
    models = [
    {
        "name": "static_random_forest",
        # 100 stromov; n_jobs=-1 využíva všetky dostupné jadrá CPU
        "model": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "save_path": STATIC_RF_METRICS_PATH,
    },
    {
        "name": "static_logistic_regression",
        "model": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        ),
        "save_path": STATIC_LR_METRICS_PATH,
    },
]

    for experiment in models:
        model_name = experiment["name"]
        model = experiment["model"]
        save_path = experiment["save_path"]

        # Jednorazový tréning na trénovacej množine (TRAIN_SIZE vzoriek)
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        # Kumulatívne vyhodnotenie na testovacej množine
        evaluate_static_model_over_time(
            model=model,
            model_name=model_name,
            X_test=X_test,
            y_test=y_test,
            test_dates=test_dates,
            save_path=save_path,
            save_every=save_every
        )


if __name__ == "__main__":
    run_static_experiments(
        #nrows=100_000,
        save_every=500
    )