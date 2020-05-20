# import flask
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# import mode
import keras
from keras_transformer import get_custom_objects, get_model, decode
model = keras.models.load_model('./saved_model.hdf5', custom_objects=get_custom_objects())

import numpy as np
#load vocab
#read dict
import json
with open('inp_dict.json', 'r') as inp_d:
    inp_dict = json.load(inp_d)
with open('tar_dict.json', 'r') as tar_d:
    tar_dict = json.load(tar_d)
with open('cst.json', 'r') as const:
    cst = json.load(const)

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


# sanity check route

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    name = [x for x in request.form.values()]
    print('POST')
    print(name[0])
    inp_txt = name[0].strip()
    input_tokens = [[i for i in inp_txt]]
    inp_encoded = [['<START>'] + tokens + ['<END>'] for tokens in input_tokens]
    inp_max_len = 40
    inp_padded = [tokens + ['<PAD>'] * (inp_max_len - len(tokens)) for tokens in inp_encoded]
    encode_inp = [list(map(lambda x: inp_dict[x], tokens)) for tokens in inp_padded]
    decoded = decode(
        model,
        encode_inp,
        start_token=1,
        end_token=2,
        pad_token=0,
    )
    tar_dict_inv={v: k for k, v in tar_dict.items()}
    answer= ''.join(map(lambda x: tar_dict_inv[x], decoded[0][1:-1]))
    print(answer)
    return render_template('home.html', prediction_text='output_name {}'.format(name[0]))


if __name__ == '__main__':
    app.run()
