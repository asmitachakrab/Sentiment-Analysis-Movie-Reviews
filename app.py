import pickle
import os
import streamlit as st

# Define the relative path to the model
MODEL_PATH = os.path.join('models', 'sentiment_model.pkl')

# Load the model
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    st.success("✅ Model loaded successfully.")
except FileNotFoundError:
    st.error("❌ Model file not found. Please make sure 'models/sentiment_model.pkl' exists.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Streamlit UI for the movie review sentiment analyzer
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to analyze its sentiment. The model will predict if the review is **positive** or **negative**.")

# User input for movie review
review = st.text_area("📝 Enter a movie review:")

# Button to analyze the review
if st.button('Analyze Sentiment'):
    if review:
        # Make prediction
        prediction = model.predict([review])[0]

        # Adjust this if your model returns labels like strings or different integers
        if prediction == 1 or prediction == 'pos':
            sentiment = "Positive 😊"
        else:
            sentiment = "Negative 😞"

        st.write(f"🔍 Sentiment: {sentiment}")
    else:
        st.warning("⚠️ Please enter a review.")
