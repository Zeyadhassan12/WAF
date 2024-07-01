from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load the trained SVM model
model_path = '/Users/zeyadhassan/Desktop/WAF/svm_model-2.pkl'
vectorizer_path = '/Users/zeyadhassan/Desktop/WAF/tfidf_vectorizer.pkl'
bert_model_path = '/Users/zeyadhassan/Desktop/WAF/bert_model'
bert_tokenizer_path = '/Users/zeyadhassan/Desktop/WAF/bert_tokenizer'

try:
    with open(model_path, 'rb') as f:
        svm_model = pickle.load(f)
    print("SVM model loaded successfully.")
except Exception as e:
    svm_model = None
    print(f"Error loading SVM model: {e}")

try:
    with open(vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    print("TF-IDF vectorizer loaded successfully.")
except Exception as e:
    tfidf_vectorizer = None
    print(f"Error loading TF-IDF vectorizer: {e}")

try:
    bert_model = TFBertForSequenceClassification.from_pretrained(bert_model_path)
    tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
    print("BERT model and tokenizer loaded successfully.")
except Exception as e:
    bert_model = None
    print(f"Error loading BERT model: {e}")

@app.route('/')
def index():
    return render_template('index11.html')  # Ensure this template exists in the 'templates' folder

def get_svm_probabilities(input_text):
    input_tfidf = tfidf_vectorizer.transform([input_text])
    probabilities = svm_model.predict_proba(input_tfidf)[0]
    return probabilities

def get_bert_probabilities(input_text):
    inputs = tokenizer(input_text, return_tensors='tf', padding=True, truncation=True)
    outputs = bert_model(**inputs)
    probabilities = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    return probabilities

def get_combined_prediction(input_text):
    svm_prob = get_svm_probabilities(input_text)
    bert_prob = get_bert_probabilities(input_text)
    
    # Ensure both probability arrays are of the same shape
    if len(svm_prob) != len(bert_prob):
        raise ValueError(f"Shape mismatch: SVM probabilities {len(svm_prob)} vs BERT probabilities {len(bert_prob)}")

    # Average the probabilities
    average_prob = (svm_prob + bert_prob) / 2

    # Get the final prediction
    final_prediction = np.argmax(average_prob)
    final_label = "anomaly" if final_prediction == 0 else "normal"

    return final_label, average_prob

@app.route('/simulate_attack', methods=['POST'])
def simulate_attack():
    global svm_model, tfidf_vectorizer, bert_model

    payload = request.json.get('payload')
    if not payload:
        return jsonify({"error": "No payload provided."}), 400

    try:
        print(f"Received payload: {payload}")
        
        # Preprocess payload for prediction
        payload_tfidf = tfidf_vectorizer.transform([payload])
        print(f"Payload TF-IDF shape: {payload_tfidf.shape}")

        # Predict using the SVM model
        svm_probabilities = svm_model.predict_proba(payload_tfidf)
        svm_anomaly_probability = svm_probabilities[0][1]

        # Preprocess payload for BERT prediction
        bert_probabilities = get_bert_probabilities(payload)
        bert_anomaly_probability = bert_probabilities[0]  # Assuming BERT returns probabilities directly

        # Combine predictions using average probabilities
        combined_probability = (svm_anomaly_probability + bert_anomaly_probability) / 2
        
        threshold = 1  # Set your threshold for anomaly detection

        if combined_probability > threshold:
            return jsonify({"result": "Potential SQL Injection detected."}), 403
        else:
            return jsonify({"result": "No SQL Injection detected."}), 200

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



