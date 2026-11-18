# 🐦 Twitter Sentiment Analysis

**Classification automatique de tweets de compagnies aériennes (positif / neutre / négatif)**
**NLP · feature engineering · TF-IDF mots + caractères · régression logistique · évaluation et interprétation**

---

## 📌 Description du projet

Ce projet analyse **14 604 tweets** adressés à six compagnies aériennes américaines (février 2015) et prédit leur **sentiment** :

➡️ **negative**
➡️ **neutral**
➡️ **positive**

Le livrable est un **notebook Jupyter unique**, qui couvre la chaîne complète :

- ✔️ Analyse exploratoire commentée (8 axes, 10 figures)
- ✔️ Nettoyage NLP respectueux des négations
- ✔️ **Feature engineering** : 22 variables linguistiques + TF-IDF mots (1-2 grammes) + TF-IDF caractères (3-5 grammes)
- ✔️ Protocole sans fuite de données (vectorisation ajustée sur le train uniquement)
- ✔️ Comparaison de 5 modèles + baseline, validation croisée stratifiée 5-fold, réglage des hyperparamètres
- ✔️ Évaluation sur un jeu de test tenu à l'écart (2 921 tweets)
- ✔️ **Interprétation** : ablation des features, coefficients par classe, analyse d'erreurs, prédiction sur tweets inédits
- ✔️ Sauvegarde du pipeline complet (`sentiment_model.joblib`)

Dataset : [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment/data) (CrowdFlower, 2015).

---

## 📂 Structure du dépôt

| Fichier | Rôle |
|---|---|
| `Twitter_Sentiment_Analysis.ipynb` | **Le notebook principal**, (34 cellules de code, 21 figures, tableaux de résultats) |
| `dataset.csv` | Dataset Kaggle (14 640 lignes d'origine) — à placer à côté du notebook |
| `sentiment_model.joblib` | Pipeline final sauvegardé (~1,8 Mo), généré en fin de notebook |
| `sentiment-cleaner-classifier.ipynb` | Ancienne version du notebook, conservée pour historique (non utilisée) |

---

## 📊 Données

Après suppression des **36 doublons stricts**, le corpus contient **14 604 tweets** répartis ainsi :

| Sentiment | Effectif | Part |
|---|---:|---:|
| negative | 9 159 | 62,7 % |
| neutral | 3 091 | 21,2 % |
| positive | 2 354 | 16,1 % |

Ratio de déséquilibre : **3,9 tweets négatifs pour 1 positif**.

Colonnes principales :

| Colonne | Description | Remarque |
|---|---|---|
| `text` | Contenu du tweet | seule variable d'entrée du modèle |
| `airline_sentiment` | Label d'origine (3 classes) | variable cible |
| `airline` | Compagnie mentionnée | 6 compagnies, exclue du modèle (a priori de marque) |
| `negativereason` | Motif de plainte | 37,3 % de manquants = **tous** les tweets non négatifs → exclue (fuite de cible) |
| `retweet_count` | Nombre de retweets | 94,8 % de zéros → écartée |
| `tweet_location`, `user_timezone` | Profil utilisateur | ~33 % de manquants, biais déclaratif → descriptif seulement |
| `airline_sentiment_confidence` | Confiance de l'annotation | moyenne 0,90 ; 236 tweets < 0,50 → bruit de labels |

Encodage de la cible :

```
negative → 0
neutral  → 1
positive → 2
```

**Période couverte :** 16 → 24 février 2015 (heure du Pacifique), avec un pic à **3 079 tweets le 22 février** (≈ 2,2× la médiane), signature d'un incident opérationnel majeur.

---

## 🔍 Analyse exploratoire (résumé)

Chaque figure du notebook est suivie d'une interprétation chiffrée. Points saillants :

- **Compagnies** : US Airways 77,7 % de tweets négatifs, American 71,3 %, United 68,9 %, contre **Virgin America 35,9 %** et Delta 43,0 %.
- **Motifs de plainte** : service client (31,7 %), vol en retard (18,1 %), annulation (9,2 %), bagages perdus (7,9 %) ; 13,0 % de « Can't Tell » (plainte sans cause identifiable).
- **Longueur** : les tweets négatifs font 113,9 caractères / 19,7 mots en moyenne, contre 87,4 / 14,4 pour les neutres et 86,1 / 14,0 pour les positifs → **+40 % de mots**.
- **Expressivité** : 0,84 « ! » par tweet positif contre 0,29 par tweet négatif (contre-intuitif, mais mesuré).
- **Viralité** : 94,8 % des tweets ne sont jamais retweetés (maximum 44).
- **Vocabulaire** : *cancelled, hold, delay, customer, bag* (négatif) vs *thanks, great, love, awesome* (positif) vs *please, help, need, would* (neutre).

---

## 🧹 Nettoyage du texte

Fonction utilisée : `clean_text` (§5.2 du notebook), appliquée **vectoriellement** (`Series.map`) et non par boucle indexée.

Étapes :

1. minuscules, décodage HTML, URLs remplacées par un jeton `url`
2. suppression des mentions `@compagnie` (leur nombre devient une variable)
3. hashtags ramenés au mot nu
4. réduction des répétitions (`sooooo` → `soo`)
5. **développement des contractions mot par mot** (`don't` → `do not`, `you're` → `you are`) — jamais en sous-chaîne
6. suppression des stopwords **en conservant les négations** et les intensificateurs (`not`, `never`, `no`, `too`, `very`, `but`…)
7. lemmatisation WordNet avec cache mémoire

Exemple réel produit par le notebook :

| Tweet original | Après nettoyage |
|---|---|
| `@united our flight was delayed 3 hours, NOT happy at all!!!` | `flight delay hour not happy` |
| `Thanks @SouthwestAir, the crew was sooooo friendly :)` | `thank crew soo friendly` |
| `Don't fly @AmericanAir, they never help when bags are lost.` | `not fly never help bag lose` |
| `Is there a shuttle to the terminal?` | `shuttle terminal` |

> Pas de stemming agressif (contrairement à l'ancienne version) : la lemmatisation préserve la lisibilité des coefficients, et les n-grammes de caractères couvrent les variantes morphologiques.

---

## 🛠️ Feature engineering

### 1. Variables linguistiques (22)

| Bloc | Variables |
|---|---|
| Taille / richesse | `n_chars`, `n_words`, `avg_word_len`, `n_unique_words`, `lexical_diversity` |
| Style Twitter | `n_mentions`, `n_hashtags`, `n_urls`, `n_exclam`, `n_question`, `n_ellipsis`, `n_upper_words`, `caps_ratio`, `n_digits`, `n_punct` |
| Grammaire / expressivité | `n_negations`, `has_negation`, `stopword_ratio` |
| Polarité lexicale (VADER) | `vader_neg`, `vader_neu`, `vader_pos`, `vader_compound` |

Pouvoir discriminant mesuré dans le notebook :

- `vader_compound` : corrélation de Spearman **+0,49** avec la cible, information mutuelle **0,225 nat** (la plus informative)
- `vader_neg` (−0,43), `vader_pos` (+0,41)
- longueur : `n_words` (−0,36), `n_unique_words` (−0,37), `n_chars` (−0,34)
- négations : `n_negations` (−0,32)
- `caps_ratio` (+0,26) et `n_exclam` (+0,21) : corrélés **positivement** au sentiment

### 2. Vectorisation TF-IDF

- **mots + bigrammes** : 14 763 termes (`min_df=2`, `sublinear_tf=True`, accents normalisés)
- **caractères 3-5 grammes** (`char_wb`) : 15 000 sous-chaînes

Matrice finale : **29 785 features** (dont 0,09 % de valeurs non nulles sur le bloc mots).

### 3. Assemblage sans fuite

Le transformer `TweetFeatures` (classe scikit-learn) encapsule nettoyage + vectorisation + normalisation `MinMaxScaler` des 22 variables, puis concatène les blocs en matrice creuse. Il est ajusté **sur le train uniquement** et réutilisable dans un `Pipeline`.

---

## 🤖 Modélisation

Protocole :

- découpage **stratifié** 80 / 20 : 11 683 tweets d'entraînement, 2 921 de test
- **validation croisée stratifiée 5-fold sur le train** pour comparer et régler
- jeu de test touché une seule fois, à la fin
- métrique principale : **F1 macro** (le déséquilibre rend l'accuracy trompeuse)

### Comparaison des modèles

| Modèle | F1 macro (CV) | F1 macro (test) | Accuracy (test) |
|---|---:|---:|---:|
| **Régression logistique** | **0,7496** | **0,7498** | 0,7932 |
| SVM linéaire | 0,7442 | 0,7368 | 0,7936 |
| Complement NB | 0,7270 | 0,7220 | 0,7915 |
| Naive Bayes multinomial | 0,7220 | 0,7189 | 0,7929 |
| Forêt aléatoire (150 arbres) | 0,7194 | 0,7218 | 0,7877 |
| Baseline « toujours negative » | 0,2570 | 0,2570 | 0,6272 |

La baseline atteint 62,7 % d'accuracy **sans rien apprendre** : c'est pourquoi le F1 macro est mis en avant.

### Modèle final (réglé)

Recherche en grille sur `C ∈ {0,5 ; 1 ; 2 ; 4}` et `class_weight ∈ {None, balanced}` → **C = 2, class_weight = None**, soit **0,7532** de F1 macro en CV.

Performance sur le jeu de test (2 921 tweets) :

| Classe | Précision | Rappel | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0,863 | 0,910 | **0,886** | 1 832 |
| neutral | 0,666 | 0,623 | **0,644** | 618 |
| positive | 0,782 | 0,684 | **0,729** | 471 |
| **Macro** | **0,770** | **0,739** | **0,753** | 2 921 |

**Accuracy : 0,813** · F1 pondéré : 0,809.

Matrice de confusion : `[[1667, 121, 44], [187, 385, 46], [77, 72, 322]]` (lignes = réalité).

---

## 🧪 Étude d'ablation : ce que le feature engineering apporte

Même classifieur (régression logistique), même protocole, seule la représentation change.

| Représentation | Nb features | F1 macro (CV) | F1 macro (test) |
|---|---:|---:|---:|
| Sac de mots (1-grammes) | 3 905 | 0,7130 | 0,7119 |
| TF-IDF mots (1-2 grammes) | 14 763 | 0,7280 | 0,7257 |
| TF-IDF mots + caractères | 29 763 | 0,7324 | 0,7344 |
| Variables engineered seules | 22 | 0,6495 | 0,6527 |
| **TF-IDF + variables (complet)** | **29 785** | **0,7496** | **0,7498** |

Lecture :

- le **texte reste le moteur** ; 22 variables seules plafonnent à 0,650
- TF-IDF apporte **+1,5 point** sur le sac de mots ; les n-grammes de caractères **+0,4 point** (robustesse aux fautes et abréviations)
- les variables linguistiques ajoutent **+1,7 point** au texte seul : un rôle de **correctif** (négation, intensité, longueur), pas de moteur
- gain total vs baseline majoritaire : **+0,493 de F1 macro**

---

## 🔎 Interprétation des résultats

### Coefficients les plus influents (modèle final)

| Classe | Features (coefficient) |
|---|---|
| positive | `n_exclam` (+3,02), `not wait` (+2,46 — bigramme de « can't wait »), `vader_compound` (+2,26), `no worry` (+1,80), *great* (+1,67), *crew* (+1,35) |
| negative | `vader_neg` (+2,21), *hour* (+2,16), *hr* (+2,12), *hold* (+1,51), *bag* (+1,51), *delay* (+1,45), *min* (+1,43) |
| neutral | `n_question` (+4,57), `delay cancel` (+1,85), *hi* (+1,66), `would like` (+1,49), `caps_ratio` (+1,32), `n_urls` (+1,23) |

Le modèle a appris que **la durée chiffrée** (*hour*, *hr*, *min*) est un marqueur central de la plainte — un signal invisible pour un lexique de sentiment classique.

### Analyse d'erreurs (547 erreurs, 18,7 %)

| Réel → prédit | Cas |
|---|---:|
| neutral → negative | 187 |
| negative → neutral | 121 |
| positive → negative | 77 |
| positive → neutral | 72 |
| neutral → positive | 46 |
| negative → positive | 44 |

- La frontière **negative ↔ neutral** concentre l'essentiel des erreurs : un tweet comme *« @united what is the holdup? »* est une question (neutre) écrite avec le vocabulaire de la plainte.
- Une partie des erreurs `positive` provient du **bruit d'étiquetage** (plaintes annotées `positive` par la foule), pas du modèle.
- Le **sarcasme** reste une limite : *« Oh great, delayed again. Amazing service as always… »* est classé `positive` avec **0,962** de confiance.

### Prédiction sur de nouveaux tweets

Fonction `predict_sentiment(texts)` (texte brut, pipeline complet) :

| Tweet | Prédiction | Confiance |
|---|---|---:|
| `I love this flight so much, thank you for the amazing crew!` | positive | 0,998 |
| `I hate this flight, three hours late and no information.` | negative | 0,999 |
| `@united cancelled our flight 4 hours ago and nobody at the counter can help us #neveragain` | negative | 0,980 |
| `Oh great, delayed again. Amazing service as always...` | positive ❌ | 0,962 |
| `Is there a shuttle from terminal B to terminal C?` | neutral | 0,811 |

---

## 📈 Visualisations incluses (21 figures)

- **Qualité** : taux de valeurs manquantes par colonne
- **Cible** : barplot + camembert des sentiments
- **Compagnies** : heatmap des parts par compagnie + volumes
- **Motifs de plainte** : barplot horizontal annoté
- **Temporalité** : volume quotidien (avec pic annoté) + volume horaire par sentiment
- **Longueur / viralité** : histogrammes par sentiment, boxplot, distribution des retweets
- **Géographie** : top fuseaux horaires et localisations
- **Vocabulaire** : top 15 mots par classe + **4 nuages de mots** (global, négatif, neutre, positif)
- **Features** : corrélations de Spearman, information mutuelle, boxplots des variables clés
- **Modélisation** : top 20 termes χ², boxplot de stabilité des modèles, matrices de confusion (effectifs et normalisées), ablation, coefficients par classe, heatmap des coefficients des variables, probabilités sur de nouveaux tweets

Chaque figure est suivie d'une cellule **« Interprétation »** en markdown.

---

## 🚀 Technologies utilisées

- Python 3.12
- pandas, numpy, scipy
- scikit-learn (`Pipeline`, `TfidfVectorizer`, `LogisticRegression`, `LinearSVC`, `MultinomialNB`, `ComplementNB`, `RandomForestClassifier`, `GridSearchCV`)
- NLTK (stopwords, WordNet, **VADER**)
- matplotlib, seaborn, wordcloud
- joblib

Versions testées lors de l'exécution : `pandas 3.1.1`, `numpy 2.5.5`, `scikit-learn 1.9.1`, `matplotlib 3.11.2`, `seaborn 0.13.2`, `nltk 3.10.4`, `wordcloud 1.9.6`.

---

## ▶️ Lancer le projet

### 1. Installer les dépendances

```bash
pip install pandas numpy scipy scikit-learn nltk matplotlib seaborn wordcloud joblib jupyter
```

> Aucun `requirements.txt` n'est fourni : le notebook liste ses imports dans la cellule « Environnement de travail ». Les ressources NLTK (`stopwords`, `wordnet`, `vader_lexicon`, `omw-1.4`) sont téléchargées automatiquement au démarrage (connexion requise la première fois).

### 2. Placer le dataset

Télécharger `dataset.csv` depuis [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment/data) et le placer **à côté du notebook**. Un message d'erreur explicite s'affiche si le fichier est absent.

### 3. Exécuter

```bash
jupyter lab        # ou : jupyter notebook
```

Ouvrir `Twitter_Sentiment_Analysis.ipynb` puis **Run All**. Durée typique : 4 à 6 minutes sur 2 cœurs (la forêt aléatoire et la recherche en grille sont les postes les plus coûteux).

Le notebook est déjà livré **avec ses sorties** : il est lisible sans réexécution.

### 4. Recharger le modèle sauvegardé

```python
import joblib
import pandas as pd

model = joblib.load("sentiment_model.joblib")
model.predict(pd.Series(["The flight was amazing!"]))
```

> ⚠️ Le pipeline contient le transformer `TweetFeatures`, qui référence `clean_text` et `extract_meta_features`. Pour le recharger dans une nouvelle session, **exécutez d'abord les cellules §5.2, §5.3 et §6.1** du notebook (ou déplacez ces définitions dans un module `features.py` importé des deux côtés). C'est la bonne pratique pour un usage en production.

---

## ⚠️ Limites

1. **Fenêtre courte** (9 jours, février 2015) et **un seul secteur** : pas de garantie de généralisation à d'autres périodes, compagnies ou langues.
2. **Labels bruités** (annotation par foule) : plafond de performance estimé autour de 0,80-0,85 de F1 macro.
3. **Sac de mots / TF-IDF** : pas de compréhension du sarcasme, de l'ironie ni des négations imbriquées ; les n-grammes et VADER atténuent sans résoudre.
4. **Pas d'analyse de cause** : le modèle prédit un sentiment, pas un motif de plainte.
5. **Biais de collecte** : uniquement les tweets mentionnant les compagnies, profils géolocalisés à 67 %.

---

## ✨ Améliorations futures

- **Transformers** : fine-tuning DistilBERT / RoBERTa (gain attendu de plusieurs points de F1 macro)
- **Classification hiérarchique** : sentiment puis motif de plainte (multi-tâches)
- **Calibration** des probabilités et seuils ajustés au coût métier
- **Apprentissage actif** sur les tweets à faible confiance et les « Can't Tell »
- **Industrialisation** : API de prédiction, dashboard Streamlit, surveillance de la dérive, réentraînement périodique

---

## 🔁 Évolutions par rapport à la version précédente

| Ancien README | Version actuelle |
|---|---|
| Random Forest comme modèle principal, accuracy ~87 % | **Régression logistique** sur TF-IDF + variables ; **F1 macro 0,753**, accuracy 0,813 ; RF reléguée (0,722) après comparaison |
| TextBlob | **VADER** (NLTK) + TF-IDF mots **et** caractères |
| Imputation KNN + OneHotEncoding de variables | Supprimées : `negativereason`/`airline`/date sont écartées pour éviter la **fuite de cible** ; 22 variables linguistiques remplacent l'imputation |
| Stemming Lancaster + lemmatisation | **Lemmatisation seule**, négations conservées, contractions développées mot par mot |
| `clean_text` par boucle sur `range(14640)` | Nettoyage vectorisé, sans index codé en dur |
| Scores par classe 0,95 / 0,76 / 0,67 | **0,886 / 0,644 / 0,729** (mesurés sur le test, modèle final) |
| Scripts `sentiment_analysis.py` / `predict.py` | **Notebook unique** + `sentiment_model.joblib` |
| Importance des features RF | **Coefficients par classe**, ablation chiffrée, analyse d'erreurs détaillée |

---

## 👤 Auteur

**Alex Alkhatib**
Projet NLP — Classification de tweets par sentiment

---

## 📄 Licence

MIT License
Copyright (c) 2025 Alex Alkhatib
