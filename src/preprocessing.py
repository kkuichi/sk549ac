import re


def preprocess_tweet(text):
    """
    Predspracuje text tweetu pre vektorizáciu pomocou GloVe embeddingov.

    Parametre:
        text (str): Surový text tweetu.

    Návratová hodnota:
        str: Predspracovaný text tweetu.
    """

    # Prevod na malé písmená
    text = text.lower()
    
    # Odstránenie URL adries
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # Odstránenie použivateľských zmienok (@mentions)
    text = re.sub(r"@\w+", "", text)
    
    # Odstránenie znaku #pri hashtagoch, ostane iba samotné slovo
    text = re.sub(r"#(\w+)", r"\1", text)
    
    # Redukcia viacnasobne opakujúcich sa znakov na max 2
    text = reduce_repeated_chars(text, max_repeats=2)
    
    # Ponechanie iba písmen, číslic, apostrofov a medzier
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    
    # Odstránenie nadbytočných medzier a orezanie okrajov reťazca
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def reduce_repeated_chars(text, max_repeats=2):
    """
    Redukuje viacnásobne opakujúce sa znaky na zadaný maximálny počet výskytov.

    Parametre:
        text (str): Vstupný text.
        max_repeats (int): Maximálny povolený počet opakovaní znaku (predvolene 2).

    Návratová hodnota:
        str: Text s redukovanými opakujúcimi sa znakmi.

    Príklad:
        awwwwww -> aww
        soooo   -> soo
    """
    pattern = r"(.)\1{" + str(max_repeats) + r",}"
    return re.sub(pattern, r"\1" * max_repeats, text)