from flask import Flask, render_template, Response, jsonify
from camera import Video
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
class PredictionObject:
    image = []
    predict_result = []

myPrediction = PredictionObject()

@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame, predictions = camera.get_frame()
        myPrediction.predict_result = predictions
        # print(mtx)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')


@app.route('/chart_data')
def chart_data():
    x = np.array(myPrediction.predict_result)
    response = jsonify(result=x.tolist())
    response.headers.add('Access-Control-Allow-Origin', '*')
    # print(x)
    return response


@app.route('/video')
def video():
    return Response(gen(Video()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
