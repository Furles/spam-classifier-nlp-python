import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from imblearn.over_sampling import SMOTE
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

df = pd.read_csv("spam.csv", encoding="latin-1")

print(df.head())

print(df.columns)

df = df[["v1", "v2"]]

df.columns = ["label", "message"]

print(df.columns)

print(df.shape)

print(df.info())

print(df['label'].value_counts())

print(df.nunique())

print(df.isnull().sum())

print(df.duplicated(subset="message").sum())

print(df[df.duplicated(subset="message")].head(5))

df = df.drop_duplicates(subset="message").reset_index(drop=True)

print(df['label'].value_counts())

sns.countplot(data=df, x='label')
plt.title("Rozkład wiadomości (spam vs ham)")
plt.xlabel("Klasa")
plt.ylabel("Liczba wiadomości")
plt.show()

print(df['label'].value_counts(normalize=True) * 100)

df['length'] = df['message'].apply(len)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x='length',
    hue='label',
    element='step',
    stat='count',
    common_norm=False,
    bins=30,
    palette=['red', 'blue']
)

plt.title('Porównawczy histogram długości wiadomości dla klas spam i ham')
plt.xlabel('Długość wiadomości (liczba znaków)')
plt.ylabel('Liczba wiadomości')
plt.tight_layout()
plt.show()

spam_words = " ".join(df[df['label'] == 'spam']['message'])
ham_words = " ".join(df[df['label'] == 'ham']['message'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_words)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Najczęstsze słowa w spamie")
plt.axis('off')
plt.show()

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ham_words)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Najczęstsze słowa w klasie ham")
plt.axis('off')
plt.show()

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    word_list = nltk.word_tokenize(text)
    alphanumeric_words = [word for word in word_list if word.isalnum()]
    filtered_words = [word for word in alphanumeric_words if word not in stop_words and word not in string.punctuation]
    return " ".join(filtered_words)
    
    df['transformed_text'] = df['message'].apply(clean_text)
    
    vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df[df['label'] == 'spam']['transformed_text'])

word_counts = X.toarray().sum(axis=0)
words = vectorizer.get_feature_names_out()
top_words = sorted(zip(words, word_counts), key=lambda x: x[1], reverse=True)[:15]

plt.figure(figsize=(10, 5))
plt.bar(*zip(*top_words), color='lightcoral')
plt.xticks(rotation=45)
plt.title("Top 15 słów w spamie")
plt.xlabel("Słowa")
plt.ylabel("Liczba wystąpień")
plt.tight_layout()
plt.show()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df[df['label'] == 'ham']['transformed_text'])

word_counts = X.toarray().sum(axis=0)
words = vectorizer.get_feature_names_out()
top_words = sorted(zip(words, word_counts), key=lambda x: x[1], reverse=True)[:15]

plt.figure(figsize=(10, 5))
plt.bar(*zip(*top_words), color='lightcoral')
plt.xticks(rotation=45)
plt.title("Top 15 słów w hamie")
plt.xlabel("Słowa")
plt.ylabel("Liczba wystąpień")
plt.tight_layout()
plt.show()

df['num_characters']=df['transformed_text'].apply(len)
df['num_words']=df['transformed_text'].apply(lambda x:len(nltk.word_tokenize(x)))

print(df.describe())

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
print(df[['label', 'label_num']].head())

cv=CountVectorizer()
tfidf=TfidfVectorizer()
X = tfidf.fit_transform(df['transformed_text']).toarray()

X.shape

y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1 – Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)

# Model 2 – Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Model 3 – Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)

# Model 4 – Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

#Wyniki

print("Multinomial Naive Bayes")
print(classification_report(y_test, y_pred_mnb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mnb))

print("\nLogistic Regression")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

print("\nBernoulli Naive Bayes")
print(classification_report(y_test, y_pred_bnb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bnb))

print("\nGaussian Naive Bayes")
print(classification_report(y_test, y_pred_gnb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gnb))

models = ['MultinomialNB', 'LogisticRegression', 'BernoulliNB', 'GaussianNB']
y_preds = [y_pred_mnb, y_pred_lr, y_pred_bnb, y_pred_gnb]

metrics = {
    'Precision': [],
    'Recall': [],
    'F1-score': [],
    'Accuracy': []
}

for y_pred in y_preds:
    metrics['Precision'].append(precision_score(y_test, y_pred))
    metrics['Recall'].append(recall_score(y_test, y_pred))
    metrics['F1-score'].append(f1_score(y_test, y_pred))
    metrics['Accuracy'].append(accuracy_score(y_test, y_pred))

# Wykres
x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(12,6))
plt.bar(x - 1.5*width, metrics['Precision'], width, label='Precision')
plt.bar(x - 0.5*width, metrics['Recall'], width, label='Recall')
plt.bar(x + 0.5*width, metrics['F1-score'], width, label='F1-score')
plt.bar(x + 1.5*width, metrics['Accuracy'], width, label='Accuracy')

plt.xticks(x, models)
plt.ylim(0, 1.1)
plt.ylabel("Wartość metryki")
plt.title("Porównanie metryk modeli klasyfikacji spamu")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

model_names = ['MultinomialNB', 'LogisticRegression', 'BernoulliNB', 'GaussianNB']
predictions = [y_pred_mnb, y_pred_lr, y_pred_bnb, y_pred_gnb]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, pred, name in zip(axes.ravel(), predictions, model_names):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Macierz pomyłek – {name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.show()

cv=CountVectorizer()
tfidf=TfidfVectorizer(max_features=3500)
X = tfidf.fit_transform(df['transformed_text']).toarray()

X.shape

y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1 – Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)

# Model 2 – Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Model 3 – Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)

# Model 4 – Gaussian Naive Bayes (wymaga gęstej macierzy!)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

#Wyniki

print("Multinomial Naive Bayes")
print(classification_report(y_test, y_pred_mnb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mnb))

print("\nLogistic Regression")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

print("\nBernoulli Naive Bayes")
print(classification_report(y_test, y_pred_bnb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bnb))

print("\nGaussian Naive Bayes")
print(classification_report(y_test, y_pred_gnb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gnb))

# Inicjalizacja SMOTE
smote = SMOTE(sampling_strategy={1: 1000}, random_state=42)

# Oversampling klasy mniejszościowej
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Przed oversamplingiem: {np.bincount(y_train)}")
print(f"Po oversamplingu: {np.bincount(y_train_resampled)}")

# Trenujemy MultinomialNB na danych po oversamplingu
mnb = MultinomialNB()
mnb.fit(X_train_resampled, y_train_resampled)

bnb = BernoulliNB()
bnb.fit(X_train_resampled, y_train_resampled)
y_pred_bnb = bnb.predict(X_test)

# Predykcje na zbiorze testowym
y_pred = mnb.predict(X_test)

# Ewaluacja
print("\nMultinomial Naive Bayes")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nBernoulli Naive Bayes")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bnb))

# Nazwy podejść
models = ['Podejście 1', 'Podejście 2', 'Podejście 3']

# Wartości metryk dla klasy SPAM
precision = [1.00, 1.00, 0.98]
recall =    [0.72, 0.81, 0.89]
f1_score =  [0.84, 0.89, 0.93]

x = np.arange(len(models))  # [0, 1, 2]
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, precision, width, label='Precision', color='skyblue')
plt.bar(x,         recall,    width, label='Recall', color='salmon')
plt.bar(x + width, f1_score,  width, label='F1-score', color='lightgreen')

plt.xticks(x, models)
plt.ylim(0, 1.1)
plt.ylabel('Wartość')
plt.title('Porównanie metryk klasy SPAM dla 3 podejść')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

