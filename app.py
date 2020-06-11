from flask import Flask, render_template, redirect, request, url_for

import numpy as np
import cv2
from urllib.parse import unquote
import urllib
import base64
import re
import urllib.request

app = Flask(__name__)

@app.route('/doPredict', methods = ["POST"])
def doPredict():
    img_url = request.form.get('url')
    img_url = img_url.split(',')[1]

    decoded_img = base64.b64decode(img_url)
    with open("./test.png", "wb") as save_file:
        save_file.write(decoded_img)

    # print(np.fromstring(img, dtype = np.uint8).shape)
    # with open('./img.png', 'wb') as save_file:
    #     save_file.write(urllib.request.urlopen(img).read())

    # data = request.values['img'].split(',')[1]
    # decoded_data = base64.b64decode(data)
    # np_img = np.fromstring(decoded_data, dtype=np.uint8)

    return ""

@app.route('/')
def template():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000)
    app.debug = True