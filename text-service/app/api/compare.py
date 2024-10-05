import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymorphy2 import MorphAnalyzer
import pandas as pd

def text_prepare(text):
    morph = MorphAnalyzer()
    #токенизация
    tokens = word_tokenize(text)
    stop_words = stopwords.words('russian')
    punctuation_marks = (',', '.', '!', '?', ';', ':', '-', '—', '[', ']', '{', '}', "«", '»')
    #лемматизация
    tokens = [morph.normal_forms(token)[0] for token in tokens]
    #исключение стоп-слов и знаков препинания
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if token not in punctuation_marks]
    df = pd.DataFrame(tokens)
    return df

 
def text_similarity(text1, text2):
    df1 = text_prepare(text1)
    df2 = text_prepare(text2)
    df1 = df1.apply(' '.join)
    df2 = df2.apply(' '.join)
    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([df1[0], df2[0]])
    corr_matrix = tfidf_matrix * tfidf_matrix.T    
    return corr_matrix