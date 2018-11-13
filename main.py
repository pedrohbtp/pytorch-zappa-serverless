from flask import Flask
from flask_cors import CORS, cross_origin
from flask import Response
import flask
import json
import pickle
import os
import requests
# from fastai.text import get_language_model
from pytorch_models import awd_lstm
import numpy as np
import torch 
import boto3

app = Flask(__name__)
CORS(app)

MODEL_PATH= 'models/awd_lstm/lstm_wt103.pth'
MODEL_PATH_PICKLE= 'models/awd_lstm/lstm_wt103.pkl'
ITOS_PATH = 'models/awd_lstm/itos_wt103.pkl'

def pickle_obj(data, path):
    with open(path,'wb+') as f:
        pickle.dump(data,f)

def unpickle_obj(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def unpickle_s3_obj(bucket, path):
    if IS_LOCAL == False:
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, path)
        body_string = obj.get()['Body'].read()
        data = pickle.loads(body_string)
    else:
        print('is local: ', IS_LOCAL)
        data = unpickle_obj(path)
    return data
    
def sample_model(model, input_words, l=50):
    '''
    model: pytorch LM 
    s: list of strings
    '''
    no_space_words = ["'s", "'ll", ",", "?",".", "'t", "'m", "n't", "!", "'", "'ve", ";"]
    capitalize_words = ['.', '!', '\n']
    exclude_tokens = [model.stoi[i] for i in ['"',"[", "]", "(", ")", "xxup", "xxfld", "xxrep", "xxbos"] if i in model.stoi]
    bs = model[0].bs
    model[0].bs=1
    model.eval()
    model.reset()
    final_string = ''
    # Gives the model the input strings
    for s in input_words:
        t = model.stoi[s] if s in model.stoi else 0 
        res,*_ = model.forward(torch.tensor([[t]]).cpu())
        final_string = final_string + ' ' + s
    last_word = None
    # predicts l number of next words
    for i in range(l):
        result_indexes = torch.multinomial(res[-1].exp(), 10)
        # selects a word that is not in the exclude_tokens list
        for r in result_indexes:
            if r != 0 and r not in exclude_tokens: 
                word_index = r
                break
            else:
                word_index = result_indexes[0]
        
        word = model.itos[word_index]
        res, *_ = model.forward(torch.tensor([[word_index.tolist()]]).cpu())
        # Capitalize if it is the last word in a phrase
        word = word.capitalize() if last_word in  capitalize_words else word
        # Do not place a space in front of the words in no_space_words
        if word in no_space_words:
            final_string = final_string + word
        else: 
            final_string = final_string + ' ' + word
        last_word = word
    model[0].bs=bs
    return final_string

def unpickle_predict():
    model = unpickle_s3_obj(os.environ["models_bucket"], MODEL_PATH_PICKLE)
    return {'text': sample_model(model, [''], l=50)}

def load_lm_and_predict():
    ''' input_data: data to be used in the prediction
    model_path: path to the 
    '''
    #Unpickles list representing vocabulary
    itos = unpickle_obj(ITOS_PATH)
    # Generates dictionary mapping token to int
    stoi = {i[1]:i[0] for i in enumerate(itos)}
    # Generates AWD_LSTM model
    dps = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 1.0
    my_model = awd_lstm.get_language_model(vocab_sz=len(itos), emb_sz=400, n_hid=1150, n_layers=3, pad_token=1, input_p=dps[0],                    output_p=dps[1],weight_p=dps[2], embed_p=dps[3], hidden_p=dps[4], tie_weights=True, bias=True, qrnn=False)
    # load all the weights in the model
    my_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    my_model.itos = itos
    my_model.stoi = stoi
    #print(my_model)
    return {'text': sample_model(my_model, [''], l=50)}

@app.route('/inference',methods=['GET'])
def inference():
    ''' 
    GET: Performs inference on the language model
    '''
    #form_data = flask.request.get_json()
    #data = form_data['data']
    # response = unpickle_predict()
    response = load_lm_and_predict()
    resp = Response(response=json.dumps({"response": response}), status=200, mimetype='application/json')
    return resp


IS_LOCAL = False
if __name__ == '__main__':
    IS_LOCAL = True
    env = 'dev'
    # Test if the values were set to know if it is running locally or on lambda
    json_data = open('zappa_settings.json')
    env_vars = json.load(json_data)
    for key, val in env_vars[env]['aws_environment_variables'].items():
        os.environ[key] = val
    print('Set the environ')
    app.run(debug=True, host='0.0.0.0', port=8082)