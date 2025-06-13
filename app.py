import streamlit as st
import numpy as np
import pickle
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = 'wiki_model.h5'
TOKENIZER_PATH = 'tokenizerr_wiki.pickle'

# Google Drive file ID for your model
MODEL_FILE_ID = '1x4Quqg-Jo_NgB4DA1wAnD687dJL2_A7b'

# Download model if not present
if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={MODEL_FILE_ID}', MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Predict next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit UI
st.title("Next Word Prediction With LSTM")
input_text = st.text_input("Enter a sentence", "To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Next word: {next_word}")
