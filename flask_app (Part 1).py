from flask import Flask, render_template, request, jsonify
app = Flask(__name__)


############################################
#                                          #
#               Simple Flask               #
#                                          #
############################################

@app.route("/", methods = ["GET"])
def home():
    return "Hello World!"


@app.route("/echo", methods = ["GET", "POST"])
def echo():
    if request.method == 'GET':
        if "echo" in request.args: return request.args.get("echo")
    
    if request.method == 'POST':
        if "echo" in request.form: return request.form.get("echo")

    return "Send something to echo"

############################################
#                                          #
#             Simple Flask END             #
#                                          #
############################################

############################################
#                                          #
#               REAL AI PART               #
#                                          #
############################################

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import pickle
CWD = Path(__file__).parent.resolve()

model = tf.keras.models.load_model(CWD / "models/text_classification_model.h5")

# Load the tokenizer
with open(CWD / 'models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route("/movie-review")
def cyberbully():
    
    userText = [request.args.get('review')] # Get the user text
    new_sequences = tokenizer.texts_to_sequences(userText) # Format as a sequence
    new_padded_sequences = keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=120, padding="post", truncating="post")

    # Make prediction
    prediction = model.predict(new_padded_sequences)

    print(dict(score = prediction[0][0]))
    return jsonify(dict(score = prediction[0][0]))

############################################
#                                          #
#             REAL AI PART END             #
#                                          #
############################################

if __name__ == "__main__":
    app.run(debug = True)