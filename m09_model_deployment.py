#!/usr/bin/python

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import keras
import sys
import os

def predict_proba(url):
    reg = keras.models.load_model(os.path.dirname(__file__) + '/model.keras')

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(module_url)
    x_embeddings = embed(url.tolist()).numpy()

    return reg.predict(x_embeddings)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print('Por favor adicione la información del vehículo separada por ";"')

    else:

        url = sys.argv[1]

        p1 = predict_proba(url)

        print(url)
        print('Estimación del precio: ', p1)