# main.py

import joblib
import xgboost as xgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import malaya
import uvicorn

app = FastAPI(title="Fake News Detection API")

# Load the trained XGBoost model
try:
    model = joblib.load('xgboost_fake_news_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    raise e

# Load the Malaya Word2Vec vocab and vector
try:
    vocab, vector = malaya.wordvector.load('combine')
    print("Word2Vec model loaded successfully.")
except Exception as e:
    print("Error loading Word2Vec model:", e)
    raise e

# Define the input data model
class Headline(BaseModel):
    headline: str

# Preprocessing functions
def preprocess_text(text):
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

def get_word2vec_embeddings(text, vocab, vector, vector_size=256):
    words = text.split()  # Tokenize by spaces
    embeddings = []
    
    for word in words:
        try:
            word_index = vocab.get(word)
            if word_index is not None:
                embedding = vector[word_index]
                embeddings.append(embedding)
        except KeyError:
            continue
    
    if len(embeddings) == 0:
        return np.zeros(vector_size)
    
    return np.mean(embeddings, axis=0)

@app.post("/predict")
def predict_fake_news(headline: Headline):
    try:
        # Step 1: Preprocess the headline
        cleaned_headline = preprocess_text(headline.headline)
        
        # Step 2: Convert the headline to Word2Vec embeddings
        headline_embedding = get_word2vec_embeddings(cleaned_headline, vocab, vector)
        
        # Step 3: Prepare data for prediction (convert to DMatrix)
        headline_dmatrix = xgb.DMatrix([headline_embedding])
        
        # Step 4: Predict probabilities using the model
        probabilities = model.predict(headline_dmatrix)
        
        # Step 5: Calculate probabilities for fake and real news
        fake_probability = 1 - probabilities[0]  # Probability for fake news
        real_probability = probabilities[0]      # Probability for real news
        
        # Step 6: Determine the prediction label
        if real_probability >= 0.7:
            label = "Likely Real News"
        elif fake_probability >= 0.7:
            label = "Likely Fake News"
        else:
            label = "Uncertain"
        
        # Step 7: Return the results
        return {
            "headline": headline.headline,
            "prediction": label,
            "fake_news_probability": f"{fake_probability * 100:.2f}%",
            "real_news_probability": f"{real_probability * 100:.2f}%"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
