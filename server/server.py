import base64
import json
import os
import uuid

from flask import Flask, request, make_response
from flask_cors import CORS

from midi.output import generate_midi_from_scratch, extract_midi_node_dict
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
        encoded_str = base64.b64encode(file.read()).decode('utf-8')

    # Cleanup
    os.remove(tmp_filename)
    notes = extract_midi_node_dict(mid)

    response = {
        'base64': encoded_str,
        'notes': notes
    }

    return make_response(json.dumps(response, indent=4), 200)

