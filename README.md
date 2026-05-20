# 🐦 Twitter Sentiment Analysis

**Détection, nettoyage et classification automatique de tweets (positif / neutre / négatif)**
**Techniques : NLP, TextBlob/TF-IDF, Random Forest, visualisations**


## 📌 **Description du Projet**

Ce projet vise à analyser des tweets pour déterminer leur **sentiment** :
➡️ **Positif**
➡️ **Neutre**
➡️ **Négatif**

Le pipeline complet inclut :

- ✔️ Nettoyage et préparation des tweets
- ✔️ Prétraitement linguistique (stopwords, lemmatisation, racinisation…)
- ✔️ Vectorisation TF-IDF
- ✔️ Construction d'un pipeline sklearn
- ✔️ Classification via **RandomForestClassifier**
- ✔️ Évaluation (accuracy, f1-score…)
- ✔️ Visualisations (wordcloud, histogrammes, distributions)

Dataset utilisé :
👉 **Tweets Airlines Sentiment Dataset** (14 640 tweets)


## 📂 **Structure des Données**

Colonnes importantes :

| Colonne           | Description                                   |
| ----------------- | --------------------------------------------- |
| text              | Contenu du tweet                              |
| airline_sentiment | Label initial (positive / neutral / negative) |
| airline           | Compagnie aérienne mentionnée                 |
| retweet_count     | Nombre de retweets                            |
| negativereason    | Cause du sentiment négatif (si applicable)    |
| user_timezone     | Fuseau horaire de l’utilisateur               |

Target utilisée :

```
positive → 2  
neutral → 1  
negative → 0
```


## 🛠️ **Prétraitement des Données**

### 🔹 1. Nettoyage du texte

Chaque tweet subit :

* Mise en minuscule
* Suppression des mentions *@username*
* Suppression des URLs
* Suppression des hashtags
* Suppression des nombres
* Suppression des stopwords
* Lemmatisation
* Racinisation (stemming)
* Nettoyage des caractères spéciaux

Fonction utilisée :

```python
def clean_text(text):
    res = text.lower()
    res = re.sub("@\S+", "", res)
    res = re.sub("http[^\s]+|www\S+", "", res)
    res = res.replace("#", "")
    res = re.sub("\d+", "", res)
    res = [w for w in res.split() if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()
    res = [lemmatizer.lemmatize(w) for w in res]

    stemmer = LancasterStemmer()
    res = [stemmer.stem(w) for w in res]

    res = " ".join(res)
    return res
```


### 🔹 2. Prétraitement des variables numériques & catégorielles

- ✔️ **Numériques** → Imputation KNN + Normalisation
- ✔️ **Catégorielles** → OneHotEncoding
- ✔️ **Texte** → Pipeline TF-IDF

Pipeline :

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_variables),
        ("cat", cat_pipeline, cat_variables),
        ("text", text_pipeline, "text")
    ],
    remainder="passthrough",
    verbose=True
)
```


## 🤖 **Modèle de Classification**

Modèle principal :
➡️ **Random Forest Classifier**

```python
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
```

### 📊 **Performances obtenues**

```
Accuracy : ~87%
F1-score négatif : 0.95
F1-score neutre : 0.76
F1-score positif : 0.67
```


## 🧪 **Prédiction sur phrases personnalisées**

```python
sentences = [
    "Just touched down after an amazing flight!",
    "Flight delayed again? You're killing my schedule here.",
    "Neutral flight experience today."
]

df_sentences = pd.DataFrame({"text": sentences})
pipeline.predict(df_sentences)
```


## 📈 Visualisations Incluses

* Wordcloud des tweets positifs / négatifs
* Barplot des sentiments
* Distribution des mots
* Importance des features (TF-IDF + metadata)

Exemple d’importance des features :

| Feature                   | Importance |
| ------------------------- | ---------- |
| negativereason_confidence | 0.176      |
| text__thank               | 0.045      |
| text__flight              | 0.009      |
| airline                   | 0.008      |


## 🚀 **Technologies Utilisées**

* **Python**
* pandas
* numpy
* scikit-learn
* nltk
* TF-IDF Vectorizer
* matplotlib / seaborn
* RandomForestClassifier


## ▶️ **Lancer le Projet**

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Lancer l’analyse

```bash
python sentiment_analysis.py
```

### 3. Tester une prédiction

```bash
python predict.py --text "The flight was amazing!"
```


## ✨ Améliorations Futures

* Fine-tuning avec **Naive Bayes**, **SVM**, **Transformers**
* Lemmatisation améliorée via spacy
* Dashboard Streamlit
* Analyse temporelle des tweets
* Détection d'ironie et sarcasme


## 👤 Auteur

**Alex Alkhatib**
Projet NLP — Classification de Tweets par Sentiment


## 📄 Licence
MIT License
Copyright (c) 2025 Alex Alkhatib
