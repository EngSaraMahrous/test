import pickle
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def predict_mpg(new_text):
    ##loading the model from the saved file
    try:  
        pkl_filename = "model.pkl"
        with open(pkl_filename, 'rb') as f_in:
            model = pickle.load(f_in)
        if isinstance(new_text, dict):
            df = pd.DataFrame([new_text])  # Convert dict to DataFrame with a list to avoid ValueError
        else:
            df = pd.DataFrame([new_text], columns=['Text'])  # Assuming new_text is a string

        # Extract text data for prediction
        train_sentences = df['Text'].to_numpy()
        print(train_sentences)
        # Load or create a tokenizer (assuming it was used during training)
        tokenizer = Tokenizer(num_words=8905)  # The num_words should match the tokenizer used during training
        tokenizer.fit_on_texts(train_sentences)
        new_sequences = tokenizer.texts_to_sequences(train_sentences)
        max_length = 20  # Ensure this matches the max_length used during training
        new_padded = pad_sequences(new_sequences, maxlen=max_length, padding="post", truncating="post")
        print(new_padded)
        # Predict the class
        prediction = model.predict(new_padded)
        print(prediction)
        predicted_class = np.argmax(prediction, axis=1)

        # Map the predicted index to the actual class name
        class_names = ['Mixed', 'Negative', 'Neutral', 'Positive']  # Replace with actual class names
        predicted_label = class_names[predicted_class[0]]
    
        # Return the predicted label
        return predicted_label
    except Exception as error:
        return {'error': error}


