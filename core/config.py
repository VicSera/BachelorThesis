class Config:
    # Training
    epochs = 200
    learning_rate = 1e-3
    batch_size = 32

    log_scale_min = -16.0
    # Preprocessing
    input_directory = '../data/split'
    hop_size = 256
    num_mels = 80
    sample_rate = 16000
    n_fft = 1024
    win_length = 1024
    in_seq_len = hop_size * 4
    out_seq_len = hop_size * 4

    # Model params
    layers = 24
    stacks = 4
    residual_channels = 128
    gate_channels = 256
    skip_channels = 128
    dropout_probability = 0.0
    kernel_size = 3
    out_channels = 10 * 3

    # Upsampling
    upsample_scales = (4, 4, 4, 4)


