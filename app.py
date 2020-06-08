from flask import Flask, render_template, redirect, request, url_for

import numpy as np
import cv2
from urllib.parse import unquote
import base64
import re

app = Flask(__name__)

@app.route('/doPredict', methods = ["POST"])
def do_predict():
    data = request.values['img'].split(',')[1]
    decoded_data = base64.b64decode(data)
    np_img = np.fromstring(decoded_data, dtype=np.uint8)

    print(np_img.shape)

    return ""

@app.route('/')
def template():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000)
    app.debug = True