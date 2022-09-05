import base64
import json
import os
import uuid

from flask import Flask, request, make_response, render_template
from flask_cors import CORS
from pretty_midi import pretty_midi

from midi.output import generate_midi_from_scratch, extract_midi_node_dict, generate_midi
from midi.util import get_sequence
from model.LSTMidi import load_model

app = Flask(__name__, template_folder='template', static_folder='static')
CORS(app)

model = load_model(is_gpu=False)
app.config.from_prefixed_env()

schema = app.config['SCHEMA']
host = app.config['HOST']
port = app.config['PORT']
url = f'{schema}://{host}:{port}'

@app.route('/')
def page():
    print(url)
    return render_template('index.html', data={'url': url})


@app.route('/api/generate', methods=['POST'])
def generate():
    length = int(request.form['length'])

    if len(request.files) == 1:
        example_file = request.files['exampleFile']
        if example_file.content_type != 'audio/mid':
            return make_response('Bad file type', 400)

        source_mid = pretty_midi.PrettyMIDI(example_file)
        start_sequence = get_sequence(source_mid)
        mid = generate_midi(model, start_sequence, length)
    else:
        mid = generate_midi_from_scratch(model, length)

    tmp_filename = f'{uuid.uuid4()}.mid'
    mid.write(tmp_filename)

    with open(tmp_filename, 'rb') as file:
        encoded_str = base64.b64encode(file.read()).decode('utf-8')

    # Cleanup
    os.remove(tmp_filename)
    notes = extract_midi_node_dict(mid)

    response = {
        'base64': encoded_str,
        'notes': notes,
    }

    return make_response(json.dumps(response, indent=4), 200)

