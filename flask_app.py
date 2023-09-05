from flask import Flask, render_template, request, jsonify
import requests
import datetime
import tensorflow as tf

import git, os

from clean_text import clean_texts

from pathlib import Path
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model("text_classification_model.h5")

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Prepare new review for prediction
new_review = ["This movie is fantastic!"]
new_sequences = tokenizer.texts_to_sequences(new_review)
new_padded_sequences = pad_sequences(new_sequences, maxlen=120, padding="post", truncating="post")

# Make prediction
prediction = model.predict(new_padded_sequences)

# Interpret prediction
if np.round(prediction[0][0]) == 1:
    print("The review is positive.")
else:
    print("The review is negative.")


CWD = Path(__file__).parent.resolve()

model = tf.keras.models.load_model(CWD / "models/cyberbullying-bdlstm.h5")

with open(CWD / "models/tokenizer.json") as file:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(file.read())

app = Flask(__name__)

convos = []

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/cyberbully")
def cyberbully():
    messages = clean_texts([request.args.get("msg")], tokenizer)
    return jsonify(dict(score = model.predict(messages).T[0].tolist()[0]))

@app.route("/get")
def get():
    userText = request.args.get('msg')
    # Prepare new review for prediction
    new_review = ["This movie is fantastic!"]
    new_sequences = tokenizer.texts_to_sequences(new_review)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=120, padding="post", truncating="post")

    # Make prediction
    prediction = model.predict(new_padded_sequences)

    # Interpret prediction
    if np.round(prediction[0][0]) == 1:
        return "The review is positive."
    else:
        return "The review is negative."
    # return {"user": f"<div class='container darker'><span class='user-msg'>{userText}</span><br><span class='time-right'>{time()}</span></div>", "bot": f"<div class='container'><span>{botText}</span><br><span class='time-left'>{time()}</span></div>"}


############################################
#                                          #
#              Set Up Github               #
#                                          #
############################################

# Github webhooking
import git, os

import hmac
import hashlib

# Function to check that it is from github
def is_valid_signature(x_hub_signature, data, private_key):
    # x_hub_signature and data are from the webhook payload
    # private key is your webhook secret
    hash_algorithm, github_signature = x_hub_signature.split('=', 1)
    algorithm = hashlib.__dict__.get(hash_algorithm)
    encoded_key = bytes(private_key, 'latin-1')
    mac = hmac.new(encoded_key, msg=data, digestmod=algorithm)
    return hmac.compare_digest(mac.hexdigest(), github_signature)


@app.route('/update_server', methods=['POST'])
def webhook():
    # Github request checker
    x_hub_signature = request.headers.get('X-Hub-Signature')
    if not is_valid_signature(x_hub_signature, request.data, "secret"):
        return "YOU ARE NOT GITHUB!!"

    repo = git.Repo(CWD)
    origin = repo.remotes.origin
    origin.pull()
    return 'Updated PythonAnywhere successfully', 200

############################################
#                                          #
#            Set Up Github END             #
#                                          #
############################################

if __name__ == "__main__": 
    app.run(debug=True)
