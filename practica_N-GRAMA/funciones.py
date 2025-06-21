import nltk
import string
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def quitar_stopwords_esp(texto):
    espaniol = stopwords.words("spanish")
    return [w.lower() for w in texto if w.lower() not in espaniol
            and w not in string.punctuation
            and w not in ["'s", '|', '--', "''", "``", "-", ",-", "2025"]]

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lematizar(texto):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in texto]

def preparar_corpus(ruta_archivo, nombre_archivo):
    corpus = PlaintextCorpusReader(ruta_archivo, nombre_archivo, encoding="latin1")
    lineas = corpus.raw().splitlines()
    corpus_final = []

    for linea in lineas:
        tokens = word_tokenize(linea)
        limpio = quitar_stopwords_esp(tokens)
        lema = lematizar(limpio)
        if lema:
            corpus_final.append(" ".join(lema))
    return corpus_final

def generar_ngrama(texto, rango_ngramas=(2, 3), min_df=2):
    vectorizer = CountVectorizer(ngram_range=rango_ngramas, min_df=min_df)
    X = vectorizer.fit_transform(texto)
    suma = X.sum(axis=0)
    frecs = [(ngram, suma[0, idx]) for ngram, idx in vectorizer.vocabulary_.items()]
    frecs_ordenadas = sorted(frecs, key=lambda x: x[1], reverse=True)
    return frecs_ordenadas

def graficar_comparacion(df_bi, df_tri):
    df_comparacion = pd.concat([df_bi, df_tri])
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_comparacion, x="frecuencia", y="ngram", hue="tipo", palette="Set2")
    plt.title("Comparación de Frecuencia entre Bigramas y Trigramas")
    plt.xlabel("Frecuencia")
    plt.ylabel("N-grama")
    plt.tight_layout()
    plt.show()



# === FUNCIÓN PRINCIPAL ===

def analizar_ngrams(ruta_archivo=".", nombre_archivo="CorpusEducacion.txt", min_df=2):
    corpus_final = preparar_corpus(ruta_archivo, nombre_archivo)

    frecs_bi = generar_ngrama(corpus_final, rango_ngramas=(2, 2), min_df=2)
    frecs_tri = generar_ngrama(corpus_final, rango_ngramas=(3, 3), min_df=min_df)

    df_bi = pd.DataFrame(frecs_bi, columns=["ngram", "frecuencia"]).head(10)
    df_bi["tipo"] = "Bigrama"

    df_tri = pd.DataFrame(frecs_tri, columns=["ngram", "frecuencia"]).head(10)
    df_tri["tipo"] = "Trigrama"

    graficar_comparacion(df_bi, df_tri)