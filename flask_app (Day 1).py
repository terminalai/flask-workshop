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
CWD = Path(__file__).parent.resolve()

model = tf.keras.models.load_model(CWD / "models/cyberbullying-bdlstm.h5")

with open(CWD / "models/tokenizer.json") as file:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(file.read())

@app.route("/cyberbully")
def cyberbully():
    messages = clean_texts([request.args.get("msg")], tokenizer)
    return jsonify(dict(score = model.predict(messages).T[0].tolist()[0]))

############################################
#                                          #
#             REAL AI PART END             #
#                                          #
############################################

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
    if not is_valid_signature(x_hub_signature, request.data, w_secret):
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
    app.run(debug = True)