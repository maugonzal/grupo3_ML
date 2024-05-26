#!/usr/bin/python

import joblib
import tensorflow_hub as hub
import os
from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS

def predict_proba(url):
    reg = joblib.load(os.path.dirname(__file__) + '/model.pkl')

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(module_url)
    x_embeddings = embed(url.tolist()).numpy()

    return reg.predict(x_embeddings)


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app,
    version='1.0',
    title='API Clasificación',
    description='API para la clasificación del género de peliculas'
)

ns = api.namespace('predict', description='Estimación Género')

parser = api.parser()

parser.add_argument(
    'X',
    type=str,
    required=True,
    help='Resumen de la película',
    location='args'
)

resource_fields = api.model(
    'Resource', {'result': fields.String}
)

@ns.route('/')
class ModelApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()

        return {
            "result": predict_proba(args['X'])
        }, 200


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)