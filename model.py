import pandas as pd
import nltk
import re
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')

# Load dataset (use subset for speed)
df = pd.read_csv('zomato.csv').head(20000)

# Keep required columns
df = df[['name', 'location', 'cuisines', 'reviews_list', 'rate']]

# -------------------------------
# CLEAN RATE COLUMN (IMPORTANT)
# -------------------------------
def convert_rate(x):
    try:
        return float(str(x).split('/')[0])
    except:
        return None

df['rate'] = df['rate'].apply(convert_rate)
df = df.dropna(subset=['rate'])

# -------------------------------
# CLEAN TEXT
# -------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['cleaned_reviews'] = df['reviews_list'].apply(clean_text)

# -------------------------------
# CREATE SENTIMENT
# -------------------------------
df['sentiment'] = df['rate'].apply(lambda x: 1 if x >= 3 else 0)

# -------------------------------
# TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['cleaned_reviews'])
y = df['sentiment']

# -------------------------------
# MODEL
# -------------------------------
model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# SAVE FILES
# -------------------------------
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("✅ Model trained and saved successfully!")