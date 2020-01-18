from flask import Flask, request, Response

from Predict.flask.predictor import predict

app = Flask(__name__)


@app.route("/predictor", methods=["POST"])
def invocations():
    return Response(predict(request.json), status=200, mimetype="application/json")


app.run(host="localhost", port=8085)