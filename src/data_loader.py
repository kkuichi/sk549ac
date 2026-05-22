import pandas as pd
from config import RAW_DATA_PATH, COLUMNS


# Stĺpce načítané z datasetu (flag a user nepotrebné)
cls = ["target", "ids", "date", "text"]

def load_sentiment140(nrows=None):
    """
    Načíta dataset Sentiment140, predspracuje ho a zoradí podľa dátumu.

    Parametre:
        nrows (int / None): Počet požadovaných riadkov. None = celý dataset.

    Návratová hodnota:
        pd.DataFrame: Zoradený dataset pripravený na simuláciu dátového prúdu.
    """

    print("Loading dataset into df, sorting by date and mapping target 4->1...")
    
    # Načítanie z csv do dataframeu
    df = pd.read_csv(
        RAW_DATA_PATH,
        encoding="latin-1",
        header=None,
        names=COLUMNS,
        usecols=cls
    )

    # Mapovanie premennej sentimentu 4 (pozitívny) -> 1
    df["target"] = df["target"].replace({4: 1})

    # Prevod reťazca dátumu na typ datetime
    df["date"] = pd.to_datetime(
        df["date"],
        format="%a %b %d %H:%M:%S PDT %Y",
        errors="coerce"
    )

    # Odstránenie riadkov s nesparsovateľným dátumom
    df = df.dropna(subset=["date"])

    # Zoradenie podľa času
    df = df.sort_values("date").reset_index(drop=True)

    # Voliteľné obmedzenie počtu načítaných záznamov
    if nrows is not None:
        df = df.head(nrows).reset_index(drop=True)

    print("Done!")
    print(df.shape)
    
    return df