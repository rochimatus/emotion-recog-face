import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = keras.models.load_model('D:/belajar/python/face-web/resource/model')

labels=['angry', 'disgust', 'happy', 'fear', 'sad', 'surprise', 'neutral']

def predict_emotion(img):
    predictions = model.predict(img)
    return predictions
# def load_with_keras(path):
#     img = image.load_img(path, target_size=(128, 128))
#
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#
#     return np.vstack([x])

video = cv2.VideoCapture(0)
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:
    ret, frame = video.read()
    faces=faceDetect.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:
        x = int(x)
        y = int(y)
        x1, y1 = int(x + w), int(y + h)
        pict = frame[y:y1, x:x1]
        print(x, x1, y, y1)
        dim = (128, 128)
        print(pict.shape)
        pict = cv2.resize(pict, dim, interpolation=cv2.INTER_AREA)
        img = image.img_to_array(pict)
        img = np.expand_dims(img, axis=0)

        pict = np.vstack([img])
        predictions = predict_emotion(img=pict)
        # print(predictions)
        # print(type(x), type(y))
        mostPrediction = np.argmax(predictions)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 255), 2)
        cv2.line(frame, (x, y), (x + 50, y), (255, 255, 255), 12)
        cv2.putText(frame, labels[mostPrediction], (x, y), font, 1, (255, 255,255))
    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()