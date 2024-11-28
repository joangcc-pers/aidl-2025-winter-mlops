import logging
import pathlib

import torch
from flask import Flask, render_template, request
from torchtext.data.utils import get_tokenizer, ngrams_iterator

from model import SentimentAnalysis


VOCAB = None
MODEL = None
NGRAMS = None
TOKENIZER = None
MAP_TOKEN2IDX = None
MODEL_LOADED = False

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

# The code in this function will be executed before we recieve any request
def load_model():
    # First load into memory the variables that we will need to predict
    global VOCAB, MODEL, NGRAMS, TOKENIZER, MAP_TOKEN2IDX, MODEL_LOADED
    if not MODEL_LOADED:
        checkpoint_path = pathlib.Path(__file__).parent.absolute() / "state_dict.pt"
        checkpoint = torch.load(checkpoint_path)

        VOCAB = checkpoint["vocab"]
        embed_dim = checkpoint["embed_dim"]
        num_class = checkpoint["num_class"]
        NGRAMS = checkpoint["ngrams"]
        TOKENIZER = get_tokenizer("basic_english")
        MAP_TOKEN2IDX = VOCAB.get_stoi()

        # Inicializa el modelo y carga el estado
        MODEL = SentimentAnalysis(len(VOCAB), embed_dim, num_class)
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        MODEL.eval()

        MODEL_LOADED = True  # Marca el modelo como cargado

# Disable gradients
@torch.no_grad()
def predict_review_sentiment(text):
    # Convert text to tensor
    text = torch.tensor(
        [MAP_TOKEN2IDX[token] for token in ngrams_iterator(TOKENIZER(text), NGRAMS)]
    )

    # Compute output
    # TODO compute the output of the model. Note that you will have to give it a 0 as an offset.
    offsets = torch.tensor([0])
    output = MODEL(text, offsets)
    confidences = torch.softmax(output, dim=1)
    return confidences.squeeze()[
        1
    ].item()  # Class 1 corresponds to confidence of positive


@app.route("/predict", methods=["POST"])
def predict():
    load_model()
    """The input parameter is `review`"""
    review = request.form["review"]
    print(f"Prediction for review:\n {review}")

    result = predict_review_sentiment(review)
    return render_template("result.html", result=result)


@app.route("/", methods=["GET"])
def hello():
    """ Return an HTML. """
    load_model()
    return render_template("hello.html")


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == "__main__":
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host="localhost", port=8080, debug=True)
