import cv2
import numpy as np
from tensorflow import keras
from keras.preprocessing import image
from keras.models import load_model
model = load_model('./resource/model')
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels=['angry', 'disgust', 'happy', 'fear', 'sad', 'surprise', 'neutral']

dim = (48, 48)
face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
class Nuuumber:
    i = 0

numbee = Nuuumber()

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def predict_emotion(self, img):
        predictions = model.predict(img)
        return predictions

    def get_frame(self):
        numbee.i += 1
        ret, frame = self.video.read()
        faces = face_detect.detectMultiScale(frame, 1.3, 5)
        predictions = []
        for x, y, w, h in faces:
            x1, y1 = x+w, y+h
            pict = frame[y:y1, x:x1]
            pict = cv2.resize(pict, dim, interpolation=cv2.INTER_AREA)
            normalized = cv2.normalize(pict, None, 0, 255, cv2.NORM_MINMAX)
            grayscale = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
            img = image.img_to_array(grayscale)
            img = np.expand_dims(img, axis=0)
            pict = np.vstack([img])
            predictions = self.predict_emotion(pict)
            # print(pict)
            print(predictions)
            most_prediction = np.argmax(predictions)
            print("predict")
            # print(np.amax(predictions))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 255), 2)
            cv2.putText(frame, labels[most_prediction] + " " +str(np.amax(predictions)), (x, y), font, 1, (255, 255, 255))
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes(), predictions

