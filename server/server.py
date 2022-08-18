import base64
import os
import uuid

from flask import Flask, request, make_response
from flask_cors import CORS

from midi.output import generate_midi_from_scratch
from model.LSTMidi import load_model

app = Flask(__name__)
CORS(app)

model = load_model(is_gpu=False)


@app.route('/generate', methods=['POST'])
def generate():
    length = int(request.form['length'])
    tmp_filename = f'{uuid.uuid4()}.mid'
    mid = generate_midi_from_scratch(model, length)
    mid.write(tmp_filename)

    with open(tmp_filename, 'rb') as file:
        encoded_str = base64.b64encode(file.read())

    # Cleanup
    os.remove(tmp_filename)

    return make_response(encoded_str)

