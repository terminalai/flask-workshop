from flask import Flask, render_template, request, jsonify
app = Flask(__name__)


############################################
#                                          #
#                 Web Flask                #
#                                          #
############################################

# Reviews automatically scored with an integer from 0 to 10, indicating 0 to 5 stars
reviews = [
    dict(review = "This movie is amazing! A masterpiece by Disney", score = 10),
    dict(review = "This movie is meh! Disney has made better...", score = 5)
]

@app.route("/", methods = ["GET", "POST"])
def home():
    if request.method == "GET": return render_template("movie.html")
    
    if "review" not in request.form:
        flash("Cannot give empty review")
        return render_template("movie.html")
    
    reviews.append(dict(
        review = request.form['review'],
         score = round(perform_review([request.form['review']])[0] * 10)
    ))

    return render_template("movie.html")



@app.route("/reviews")
def review_list():
    print(reviews)
    return jsonify(reviews)
    

############################################
#                                          #
#               Web Flask END              #
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

def perform_review(messages):
    # Some preprocessing to the messages
    new_sequences = tokenizer.texts_to_sequences(messages)
    new_padded_sequences = keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=120, padding="post", truncating="post")

    # Make prediction
    prediction = model.predict(new_padded_sequences)
    return prediction[0] # Prediction given as a 1 by N matrix, where N is the number of messages, flatten to 1

@app.route("/movie-review")
def review():
    prediction = perform_review([request.args.get('review')])
    return jsonify(dict(score = float(prediction[0])))

############################################
#                                          #
#             REAL AI PART END             #
#                                          #
############################################

if __name__ == "__main__":
    app.run(debug = True)