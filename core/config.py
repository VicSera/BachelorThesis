import torch


class Config:
    INSTRUMENT = 'Acoustic Grand Piano'
    TEMPERATURE = 5

    MODEL_DIR = 'saved_models'
    MODEL_NAME = 'LSTMidi'
    SESSION = 'sesh6'
    CHECKPOINT = 'LSTMidi_Epoch5_BatchNum12000_Loss0.10604050010442734'

    DATASET_ROOT = '../data/dataset/maestro-v3.0.0'
    CSV_PATH = f'{DATASET_ROOT}/maestro-v3.0.0.csv'
    TRAINING_DATASET = f'{DATASET_ROOT}/dataset.pt'

    # -3 for Maestro, 7 for Groove
    FILENAME_INDEX = -3

    NOTES_COUNT = 128
    EXTRA_PARAMS_COUNT = 3  # Start, end, velocity
    MAX_VELOCITY = 128
    SEQUENCE_LENGTH = 50
    SEQUENCE_HOP = 1

    HIDDEN_DIM = 128
    NUM_LAYERS = 1
    DROPOUT = .2

    EPOCHS = 300
    BATCH_SIZE = 512
    LEARNING_RATE = 5e-3
    DYNAMIC_LR = False
    ANNEAL_RATE = 0.75
    ANNEAL_INTERVAL = 1000
    SAVE_INTERVAL_BATCHES = 3000

    CROSS_ENTROPY_WEIGHT = 0.05
    MSE_WEIGHT = 2
    WEIGHT_TOTAL = CROSS_ENTROPY_WEIGHT + MSE_WEIGHT


torch.set_default_dtype(torch.float32)
