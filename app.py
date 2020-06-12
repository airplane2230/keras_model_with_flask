from flask import Flask, render_template, redirect, request, url_for

import numpy as np
import cv2
from urllib.parse import unquote
import urllib
import base64
import re
import urllib.request

from tensorflow.keras.models import load_model

app = Flask(__name__)

def get_model():
    global model
    model = load_model('./saved_model/mnist_model.h5')

@app.route('/doPredict', methods = ["POST", "GET"])
def doPredict():
    global model

    img_url = request.form.get('url')
    img_url = img_url.split(',')[1]

    decoded_img = base64.b64decode(img_url)

    with open("./test2.png", "wb") as save_file:
        save_file.write(decoded_img)

    img = cv2.imread('./test2.png',cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))[..., np.newaxis]
    img = img / 255.
    img = np.expand_dims(img, 0)
    cv2.imwrite('./write.png', img)

    result = model.predict(img)

    print(np.argmax(result, axis=-1))

    return ""

@app.route('/')
def template():
    return render_template('index.html')

if __name__ == '__main__':
    get_model()
    app.run(port=5000)