# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:54:20 2020

@author: Vinayaka
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Models', 'COVID19_VGG19.h5')

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x/255.0
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Person is Infected With COVID-19 disease"
    elif preds==1:
        preds="The Person is Normal"
    else:
        preds="The Person is Infected With Viral Pneumonia"
       
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request.", 400
        f = request.files['file']
        if f.filename == '':
            return "No selected file.", 400

        try:
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            preds = model_predict(file_path, model)
            result = preds
            return result
        except Exception as e:
            # For debugging, you might want to log the error e
            return f"An error occurred processing the file: {str(e)}", 500
    return render_template('index.html') # Or redirect, or return "GET request not supported"


if __name__ == '__main__':
    app.run(debug=True)
