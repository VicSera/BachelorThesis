import json

import torch


class Config:
    INSTRUMENT = 'Acoustic Grand Piano'
    TEMPERATURE = 1

    # Training
    CHECKPOINT_ROOT = '../saved_models'
    MODEL_NAME = 'LSTMidi'
    SESSION = 'sesh17'
    CHECKPOINT_DIR = f'{CHECKPOINT_ROOT}/{MODEL_NAME}_{SESSION}'
    TRAINING_RESULTS_FILE = f'{CHECKPOINT_DIR}/results.csv'
    TRAINING_INFO_FILE = f'{CHECKPOINT_DIR}/info.json'
    RESUME_TRAINING = False
    FROM_EPOCH = 0

    # Eval
    USE_EXAMPLE = True
    SAVED_WEIGHTS_ROOT = '../best_models'
    SAVED_WEIGHTS_DIR = 'server/model'
    EVAL_INFO_FILE = f'{SAVED_WEIGHTS_DIR}/info.json'
    WEIGHT_FILE = 'LSTMidi_Epoch26'
    WEIGHTS_PATH = f'{SAVED_WEIGHTS_DIR}/{WEIGHT_FILE}'
    EXAMPLE_FILE = '../data/dataset/maestro-v3.0.0/2008/MIDI-Unprocessed_17_R1_2008_01-04_ORIG_MID--AUDIO_17_R1_2008_wav--4.midi'

    # Dataset
    DATASET_ROOT = '../data/dataset/maestro-v3.0.0'
    CSV_PATH = f'{DATASET_ROOT}/maestro-v3.0.0.csv'
    SEQUENCE_LENGTH = 50
    SEQUENCE_HOP = 1
    DATASET_DIR = f'{DATASET_ROOT}/compiled-len{SEQUENCE_LENGTH}-hop{SEQUENCE_HOP}'
    SPLITS = ['train', 'test', 'validation']
    DATASET = None

    FILENAME_INDEX = -3
    SPLIT_INDEX = -5

    NOTES_COUNT = 128
    EXTRA_PARAMS_COUNT = 3  # Start, end, velocity
    MAX_VELOCITY = 128

    # Model hyperparameters
    HIDDEN_DIM = 128
    NUM_LAYERS = 1
    DROPOUT = .2

    EPOCHS = 300
    BATCH_SIZE = 64
    OPTIMIZER = torch.optim.Adam
    OPTIMIZER_NAME = 'Adam'
    LEARNING_RATE = 5e-4
    DYNAMIC_LR = False
    ANNEAL_RATE = 0.75
    ANNEAL_INTERVAL = 10000
    SAVE_INTERVAL_BATCHES = None  # None for no intermediate saving

    # Weighted loss
    CROSS_ENTROPY_WEIGHT = 1
    MSE_WEIGHT = 1
    # WEIGHT_TOTAL = CROSS_ENTROPY_WEIGHT + MSE_WEIGHT
    WEIGHT_TOTAL = 1

    @staticmethod
    def dump_to_file():
        included_attrs = ['MODEL_NAME', 'SESSION', 'SEQUENCE_LENGTH', 'SEQUENCE_HOP', 'HIDDEN_DIM', 'NUM_LAYERS',
                          'DROPOUT', 'BATCH_SIZE', 'OPTIMIZER_NAME', 'LEARNING_RATE', 'CROSS_ENTROPY_WEIGHT',
                          'MSE_WEIGHT', 'WEIGHT_TOTAL', 'DYNAMIC_LR']

        if Config.DYNAMIC_LR:
            included_attrs += ['ANNEAL_RATE', 'ANNEAL_INTERVAL']

        attr_dict = {key: Config.__dict__[key] for key in included_attrs}
        with open(Config.TRAINING_INFO_FILE, 'w') as f:
            json.dump(attr_dict, f, indent=4)

    @staticmethod
    def load_from_file(path):
        with open(path, 'r') as f:
            attr_dict = json.load(f)
            for attr, value in attr_dict.items():
                setattr(Config, attr, value)
            print(f'Overwritten config attributes from: {path}')


Config.DATASET = {
    split: f'{Config.DATASET_DIR}/dataset_{split}.pt' for split in Config.SPLITS
}

torch.set_default_dtype(torch.float32)
