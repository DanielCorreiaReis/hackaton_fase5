# treino_modelo.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Carrega dataset de exemplos
df = pd.read_csv("dataset_componentes.csv")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["componentes"])
y = df["tipo"]

# Treina modelo
model = MultinomialNB()
model.fit(X, y)

# Salva modelo e vetorizador
joblib.dump(model, "modelo_treinado.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
