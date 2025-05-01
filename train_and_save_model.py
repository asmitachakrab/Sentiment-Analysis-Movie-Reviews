import os
import pandas as pd
import pickle
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Clean HTML tags
def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Load dataset
print("📥 Loading IMDB dataset...")
df = pd.read_csv('data/IMDB Dataset.csv')

# Clean reviews
print("🧹 Cleaning HTML tags from reviews...")
df['review'] = df['review'].apply(clean_html)

# Convert sentiments to binary labels
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Extract features and labels
texts = df['review'].values
labels = df['label'].values

# Split into training and test sets
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Create pipeline: TF-IDF + Naive Bayes
print("🧠 Training model...")
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {accuracy:.4f}")

# Save model
os.makedirs('models', exist_ok=True)
with open('models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved to 'models/sentiment_model.pkl'")
