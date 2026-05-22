import numpy as np
from gensim.models import KeyedVectors
from config import GLOVE_PATH


def load_glove_model():
    """
    Načíta predtrénované GloVe Twitter embeddingy pomocou knižnice gensim.

    Súbor GloVe neobsahuje hlavičku vo formáte word2vec,
    preto je použitý parameter no_header=True.
    """
    return KeyedVectors.load_word2vec_format(
        GLOVE_PATH,
        binary=False,
        no_header=True
    )


def tweet_to_vector(text, glove_model):
    """
    Prevedie predspracovaný text tweetu na jeden vektor
    spriemerovaním slovných embeddingov.

    Parametre:
        text (str): Predspracovaný text tweetu.
        glove_model (KeyedVectors): Načítaný model GloVe embeddingov.

    Návratová hodnota:
        Priemerný vektor tweetu. Ak sa žiadne slovo
        nenachádza v slovníku embeddingov, vráti nulový vektor.
    """
    words = text.split()
    vectors = []

    for word in words:
        # Zahrnutie slova len ak sa nachádza v slovníku
        if word in glove_model:
            vectors.append(glove_model[word])

    # Ak žiadne slovo tweetu nie je v slovníku, vráti sa nulový vektor
    if not vectors:
        return np.zeros(glove_model.vector_size, dtype=np.float32)

    # Výsledný vektor tweetu ako priemer všetkých slovných vektorov
    return np.mean(vectors, axis=0).astype(np.float32)


def vector_to_features(vector):
    """
    Prevedie numpy vektor na slovník príznakov vyžadovaný knižnicou River.

    Parametre:
        vector (np.ndarray): Vektor tweetu.

    Návratová hodnota:
        dict: Slovník v tvare {f0: hodnota, f1: hodnota, ...}.
    """
    return {f"f{i}": float(value) for i, value in enumerate(vector)}