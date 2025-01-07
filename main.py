import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz

# Load dataset
data = pd.read_csv('quora_duplicate_questions.tsv', sep='\t')

# Basic preprocessing
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

data['question1'] = data['question1'].fillna('').apply(preprocess_text)
data['question2'] = data['question2'].fillna('').apply(preprocess_text)

# Feature engineering
features = pd.DataFrame()

# Basic features
features['q1_len'] = data['question1'].apply(len)
features['q2_len'] = data['question2'].apply(len)
features['len_diff'] = abs(features['q1_len'] - features['q2_len'])
features['common_words'] = data.apply(lambda row: len(set(row['question1'].split()).intersection(set(row['question2'].split()))), axis=1)
features['word_overlap'] = features['common_words'] / (features['q1_len'] + features['q2_len'])

# Advanced features using TF-IDF
vectorizer = TfidfVectorizer()
questions = list(data['question1']) + list(data['question2'])
vectorizer.fit(questions)
q1_tfidf = vectorizer.transform(data['question1'])
q2_tfidf = vectorizer.transform(data['question2'])

# Cosine similarity feature
features['tfidf_cosine_sim'] = np.sum(q1_tfidf.multiply(q2_tfidf), axis=1)

# Fuzzy features
features['fuzzy_ratio'] = data.apply(lambda row: fuzz.ratio(row['question1'], row['question2']), axis=1)
features['fuzzy_partial_ratio'] = data.apply(lambda row: fuzz.partial_ratio(row['question1'], row['question2']), axis=1)
features['fuzzy_token_sort'] = data.apply(lambda row: fuzz.token_sort_ratio(row['question1'], row['question2']), axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, data['is_duplicate'], test_size=0.2, random_state=42)

# Model training with Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Hyperparameter tuning using GridSearch
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.2f}")

# Dimensionality reduction using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(features)

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=data['is_duplicate'])
plt.title("t-SNE Visualization of Question Pairs")
plt.show()
