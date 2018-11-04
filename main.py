from flask import Flask
from flask_cors import CORS, cross_origin
from flask import Response
import flask
import json
import pickle
import os
import requests

app = Flask(__name__)

MODEL_PATH= 'models/'

def pickle_obj(data, path):
    with open(path,'wb+') as f:
        pickle.dump(data,f)

def unpickle_obj(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data
    
def load_and_predict(input_data, model_path):
    ''' input_data: data to be used in the prediction
    model_path: path to the 
    '''
    return {'text': ""}

@app.route('/inference',methods=['POST'])
def inference():
    ''' 
    POST: Receives some parameters on the post request and makes an inference
    using the trained model
    '''
    form_data = flask.request.get_json()
    data = form_data['data']
    response = search_doc(data, model_path = MODEL_PATH)
    resp = Response(response=json.dumps({"response": response}), status=200, mimetype='application/json')
    return resp


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)
