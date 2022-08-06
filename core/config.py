class Config:
    DATASET_ROOT = '../data/dataset/groove'
    TRAINING_DATASET = f'{DATASET_ROOT}/dataset.pt'

    FILENAME_INDEX = 7

    UNIQUE_NOTES = [22, 26, 36, 37, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57, 58, 59]
    UNIQUE_NOTES_COUNT = len(UNIQUE_NOTES)
    UNIQUE_CONTROLS = [16, 17, 18, 4]
    UNIQUE_CONTROLS_COUNT = len(UNIQUE_CONTROLS)
    EXTRA_PARAMS_COUNT = 2
    SEQUENCE_LENGTH = 30

    HIDDEN_DIM = 64
    NUM_LAYERS = 3

    EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    ANNEAL_RATE = 0.75
    ANNEAL_INTERVAL = 1000


# class Config:
#     # Training
#     epochs = 200
#     batch_size = 2
#
#     learning_rate = 1e-3
#     anneal_rate = 0.75
#     anneal_interval = 1000
#
#     log_scale_min = -16.0
#     # Preprocessing
#     input_directory = '../data/split'
#     dataset_dir = '../data/dataset'
#     hop_size = 256
#     num_mels = 80
#     sample_rate = 16000
#     n_fft = 1024
#     win_length = 1024
#     # in_seq_len = hop_size * 4
#     receptive_field = 20476
#     # out_seq_len = hop_size * 4
#     out_seq_len = 64
#     in_seq_len = receptive_field + out_seq_len - 1
#
#     # Model params
#     layers = 10
#     stacks = 6
#     residual_channels = 128
#     gate_channels = 256
#     skip_channels = 128
#     dropout_probability = 0.0
#     kernel_size = 3
#     out_channels = 10 * 3
#
#     # Upsampling
#     upsample_scales = (4, 4, 4, 4)


