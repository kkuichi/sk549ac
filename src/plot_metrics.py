import os
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Cesty k výstupným CSV súborom statických modelov
STATIC_RF_METRICS_PATH = "../results/static_random_forest_metrics.csv"
STATIC_LR_METRICS_PATH = "../results/static_logistic_regression_metrics.csv"

from config import (
    ARF_METRICS_PATH,
    NB_METRICS_PATH,
    LR_METRICS_PATH,
    ARF_DRIFT_PATH,
    NB_DRIFT_PATH,
    LR_DRIFT_PATH,
)

# Zoznam metrík, pre ktoré sa generujú porovnávacie grafy
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
]

# Minimálny index vzorky, od ktorého sa zobrazujú body detekcie driftu
DRIFT_WARMUP_WINDOW = 300_000

# Maximálny rozptyl (v počte vzoriek) v rámci ktorého musia všetky modely
CONSENSUS_DRIFT_WINDOW = 10_000

# Minimálna vizuálna šírka zvýrazneného pásma driftu v grafe
MIN_HIGHLIGHT_WIDTH = timedelta(days=1)


def load_all_metrics():
    """
    Načíta CSV súbory s metrikami všetkých modelov (online aj statických)
    a zlúči ich do jedného DataFrame.

    Návratová hodnota:
        pd.DataFrame: Zlúčené metriky všetkých modelov, alebo prázdny DataFrame
                      ak žiadny súbor neexistuje.
    """
    metric_dfs = []

    metric_paths = {
        "arf": ARF_METRICS_PATH,
        "naive_bayes": NB_METRICS_PATH,
        "logistic_regression": LR_METRICS_PATH,
        "static_random_forest": STATIC_RF_METRICS_PATH,
        "static_logistic_regression": STATIC_LR_METRICS_PATH,
    }

    for model_name, path in metric_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)

            if not df.empty:
                df["model"] = model_name
                df["date"] = pd.to_datetime(df["date"])
                metric_dfs.append(df)
        else:
            print(f"Metrics file not found: {path}")

    if not metric_dfs:
        return pd.DataFrame()

    return pd.concat(metric_dfs, ignore_index=True)


def load_drift_points():
    """
    Načítanie CSV súborov so zaznamenanými bodmi detekcie driftu
    pre všetky tri online modely a zlúči ich do jedného DataFrame.

    Návratová hodnota:
        pd.DataFrame: Zlúčené body driftu, alebo prázdny DataFrame
                      ak žiadny súbor neexistuje.
    """
    drift_dfs = []

    drift_paths = {
        "arf": ARF_DRIFT_PATH,
        "naive_bayes": NB_DRIFT_PATH,
        "logistic_regression": LR_DRIFT_PATH,
    }

    for model_name, path in drift_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)

            if not df.empty:
                df["model"] = model_name
                df["date"] = pd.to_datetime(df["date"])
                drift_dfs.append(df)

    if not drift_dfs:
        return pd.DataFrame()

    return pd.concat(drift_dfs, ignore_index=True)


def filter_drift_points_after_warmup(drift_df, warmup_window=100_000):
    """
    Odfiltruje body driftu detegované pred uplynutím warmup okna.
    Detekcie v úvodnej fáze sú nespoľahlivé — model ešte nemá
    dostatočný počet vzoriek na stabilnú štatistiku.

    Parametre:
        drift_df (pd.DataFrame): DataFrame s bodmi driftu.
        warmup_window (int): Minimálny index vzorky pre zahrnutie bodu driftu.

    Návratová hodnota:
        pd.DataFrame: Filtrovaný DataFrame s bodmi driftu.
    """
    if drift_df.empty:
        return drift_df

    return drift_df[drift_df["tweet_index"] > warmup_window].copy()


def find_consensus_drift_regions(drift_df, window=50_000, required_models=None):
    """
    Nájde udalosti driftu, pri ktorých všetky požadované modely
    detegovali drift v rámci zadaného okna (rozptyl v počte vzoriek).

    Každá udalosť spotrebuje práve jeden bod za každý model. Spotrebované body
    nemôžu byť znovu použité v inej udalosti, no ostatné body v rovnakom rozsahu
    môžu slúžiť ako kotvy pre nové udalosti.

    Algoritmus (greedy, na úrovni bodov):
    - Zoradí body driftu podľa tweet_index.
    - Prechádza body od začiatku ako kotvy.
    - Preskočí už použité body.
    - Od nepoužitej kotvy i skenuje dopredu, pokiaľ je rozptyl <= window.
      Pre každý požadovaný model vyberie PRVÝ nepoužitý bod daného modelu.
    - Ak sú pokryté všetky požadované modely, označí tieto body ako použité
      a zaznamená udalosť (časový rozsah vybraných bodov).
    - Inak nepokračuje a posunie sa na ďalšiu kotvu.

    Parametre:
        drift_df (pd.DataFrame): DataFrame s bodmi driftu.
        window (int): Maximálny rozptyl v tweet_index pre konsenzuálnu udalosť.
        required_models (set | None): Množina modelov vyžadovaných pre konsenzus.
                                      None = všetky modely v DataFrame.

    Návratová hodnota:
        list: Zoznam dvojíc (start_date, end_date) pre každú konsenzuálnu udalosť.
    """
    if drift_df.empty:
        return []

    if required_models is None:
        required_models = set(drift_df["model"].unique())
    else:
        required_models = set(required_models)

    df = drift_df.sort_values("tweet_index").reset_index(drop=True)
    n = len(df)

    used = [False] * n
    events = []

    for i in range(n):
        if used[i]:
            continue

        anchor_ti = df.loc[i, "tweet_index"]
        anchor_model = df.loc[i, "model"]

        if anchor_model not in required_models:
            continue

        # Pre každý požadovaný model vyber prvý nepoužitý bod v rámci okna
        picked_per_model = {anchor_model: i}

        for j in range(i + 1, n):
            if df.loc[j, "tweet_index"] - anchor_ti > window:
                break

            if used[j]:
                continue

            m = df.loc[j, "model"]

            if m in required_models and m not in picked_per_model:
                picked_per_model[m] = j

            if required_models.issubset(picked_per_model.keys()):
                break

        # Ak nie sú pokryté všetky požadované modely, preskočí túto kotvu
        if not required_models.issubset(picked_per_model.keys()):
            continue
        
        # Označenie vybraných bodov ako použitých
        picked_indices = list(picked_per_model.values())

        for k in picked_indices:
            used[k] = True

        # Zaznamenanie časového rozsahu konsenzuálnej udalosti
        dates = [df.loc[k, "date"] for k in picked_indices]
        events.append((min(dates), max(dates)))

    return events


def plot_metric_comparison(
    df,
    drift_df,
    metric,
    save_path=None,
    tick_interval="week",
    consensus_window=CONSENSUS_DRIFT_WINDOW,
):
    """
    Vykreslí porovnávací graf kumulatívneho vývoja jednej metriky
    pre všetky modely, so zvýraznenými pásmami konsenzuálneho driftu.

    Parametre:
        df (pd.DataFrame): Zlúčené metriky všetkých modelov.
        drift_df (pd.DataFrame): Filtrované body driftu.
        metric (str): Názov metriky na zobrazenie.
        save_path (str | None): Cesta pre uloženie grafu. None = neukladá sa.
        tick_interval (str): Interval značiek na osi X — "day", "week" alebo "auto".
        consensus_window (int): Okno pre hľadanie konsenzuálnych udalostí driftu.
    """
    plt.figure(figsize=(14, 6))

    # Vykreslenie krivky metriky pre každý model
    for model_name, model_df in df.groupby("model"):
        model_df = model_df.sort_values("date")

        plt.plot(
            model_df["date"],
            model_df[metric],
            label=model_name
        )

    # Nájdenie a vykreslenie konsenzuálnych pásiem driftu
    consensus_regions = find_consensus_drift_regions(
        drift_df,
        window=consensus_window
    )

    for start_date, end_date in consensus_regions:
        # Rozšírenie príliš úzkych pásiem na minimálnu vizuálnu šírku
        if end_date - start_date < MIN_HIGHLIGHT_WIDTH:
            mid = start_date + (end_date - start_date) / 2
            start_date = mid - MIN_HIGHLIGHT_WIDTH / 2
            end_date = mid + MIN_HIGHLIGHT_WIDTH / 2

        plt.axvspan(
            start_date,
            end_date,
            color="red",
            alpha=0.15,
            linewidth=0,
        )

    # Doplnenie legendy o položku pre pásma driftu
    drift_legend_handles = []

    if consensus_regions:
        drift_legend_handles.append(
            plt.Rectangle(
                (0, 0), 1, 1,
                facecolor="red",
                alpha=0.15,
                label="concept drift (všetky modely)"
            )
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend(drift_legend_handles)

    # Nastavenie intervalu a formátu značiek na osi X
    ax = plt.gca()

    if tick_interval == "day":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    elif tick_interval == "week":
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m."))

    plt.xlabel("Dátum")
    plt.ylabel(metric)
    plt.title(f"Kumulatívne porovnanie modelov podľa metriky: {metric}")
    plt.ylim(0.55, 0.9)
    plt.legend(handles=handles)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Uloženie grafu do súboru ak je zadaná cesta
    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_all_metrics():
    metrics_df = load_all_metrics()
    drift_df = load_drift_points()

    if not drift_df.empty:
        print(f"Loaded drift points: {len(drift_df)}")

        # Odfiltrovanie bodov driftu z úvodnej warmup fázy
        drift_df = filter_drift_points_after_warmup(
            drift_df,
            warmup_window=DRIFT_WARMUP_WINDOW
        )

        print(f"Drift points after warmup filter: {len(drift_df)}")

        # Výpis nájdených konsenzuálnych udalostí driftu
        consensus_regions = find_consensus_drift_regions(
            drift_df,
            window=CONSENSUS_DRIFT_WINDOW
        )

        print(f"Consensus drift events: {len(consensus_regions)}")

        for start_date, end_date in consensus_regions:
            print(f"  - {start_date} -> {end_date}")

    # Generovanie grafu pre každú sledovanú metriku
    for metric in METRICS:
        plot_metric_comparison(
            df=metrics_df,
            drift_df=drift_df,
            metric=metric,
            save_path=f"../results/{metric}_comparison_with_drift_by_date.png",
            tick_interval="week"
        )


if __name__ == "__main__":
    plot_all_metrics()