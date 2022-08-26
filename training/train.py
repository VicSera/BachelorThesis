import os.path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config
from midi.dataset import MidiDataset
from model.LSTMidi import LSTMidi
from training.loss import custom_two_part_loss
from training.lrschedule import step_learning_rate_decay
from training.metrics import prediction_matrix_to_metrics

if __name__ == '__main__':
    model = LSTMidi().cuda()

    if Config.RESUME_TRAINING:
        print(f'Resuming training from epoch {Config.FROM_EPOCH}')
        Config.load_from_file(Config.TRAINING_INFO_FILE)
        model.load_state_dict(torch.load(f'{Config.CHECKPOINT_DIR}/{Config.MODEL_NAME}_Epoch{Config.FROM_EPOCH}'))
    else:
        if not os.path.exists(Config.CHECKPOINT_DIR):
            os.makedirs(Config.CHECKPOINT_DIR)
        Config.dump_to_file()
        print(f'Running training on {torch.cuda.get_device_name(torch.cuda.current_device())}')

        with open(Config.TRAINING_RESULTS_FILE, 'w') as f:
            f.write("epoch,loss,ce,mse,validation_loss,validation_ce,validation_mse,accuracy,precision,recall\n")

    model.train()

    optimizer = Config.OPTIMIZER(model.parameters(), lr=Config.LEARNING_RATE)

    train_dataset = MidiDataset.load(Config.DATASET['train'])
    validation_dataset = MidiDataset.load(Config.DATASET['validation'])

    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    step = Config.FROM_EPOCH * len(train_dataloader) if Config.RESUME_TRAINING else 0
    current_lr = Config.LEARNING_RATE

    epochs = range(Config.FROM_EPOCH + 1, Config.EPOCHS) if Config.RESUME_TRAINING else range(Config.EPOCHS)
    for epoch in epochs:
        # Training
        train_losses = []
        progress_bar = tqdm(enumerate(train_dataloader),
                            unit='batches',
                            total=len(train_dataloader),
                            desc=f'Training (epoch {epoch}):')
        for batch_num, (X_pitch, X_extra, pitch, extra) in progress_bar:
            step += 1

            if Config.DYNAMIC_LR:
                current_lr = step_learning_rate_decay(
                    Config.LEARNING_RATE, step, Config.ANNEAL_RATE, Config.ANNEAL_INTERVAL)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            optimizer.zero_grad()

            pitch_pred, extra_pred = model((X_pitch.cuda(), X_extra.cuda()))

            cross_entropy, mse, loss = custom_two_part_loss(pitch.cuda(), extra.cuda(), pitch_pred, extra_pred)
            loss.backward()
            optimizer.step()

            train_losses.append(torch.tensor((loss.item(), cross_entropy.item(), mse.item())))
            progress_bar.set_postfix_str(f'Loss: {loss.item()} (CE: {cross_entropy.item()}, MSE: {mse.item()}')
            if Config.SAVE_INTERVAL_BATCHES is not None and batch_num % Config.SAVE_INTERVAL_BATCHES == 0:
                checkpoint_name = f'{Config.MODEL_NAME}_Epoch{epoch}_BatchNum{batch_num}_Loss{loss.item()}'
                torch.save(model.state_dict(),
                           f'{Config.CHECKPOINT_DIR}/{checkpoint_name}')

        # Validation
        validation_losses = []
        prediction_matrix = torch.zeros((Config.NOTES_COUNT, Config.NOTES_COUNT), dtype=torch.int16)
        progress_bar = tqdm(enumerate(validation_dataloader),
                            unit='batches',
                            total=len(validation_dataloader),
                            desc=f'Validation (epoch {epoch}):')
        with torch.no_grad():
            for batch_num, (X_pitch, X_extra, pitch, extra) in progress_bar:
                step += 1
                pitch_pred, extra_pred = model((X_pitch.cuda(), X_extra.cuda()))

                cross_entropy, mse, loss = custom_two_part_loss(pitch.cuda(), extra.cuda(), pitch_pred, extra_pred)
                validation_losses.append(torch.tensor((loss.item(), cross_entropy.item(), mse.item())))

                pitch_pred_values = torch.argmax(pitch_pred, dim=-1)
                for idx in range(len(pitch)):
                    prediction_matrix[pitch_pred_values[idx], pitch[idx]] += 1
                progress_bar.set_postfix_str(f'Loss: {loss.item()} (CE: {cross_entropy.item()}, MSE: {mse.item()}')

        # Save progress after each epoch
        checkpoint_name = f'{Config.MODEL_NAME}_Epoch{epoch}'

        # Training loss
        loss, ce, mse = torch.mean(torch.stack(train_losses), dim=0).tolist()

        # Validation loss
        validation_loss, validation_ce, validation_mse = torch.mean(torch.stack(validation_losses), dim=0).tolist()

        # Metrics
        accuracy, precision, recall = prediction_matrix_to_metrics(prediction_matrix)

        info = [str(value) for value in [epoch, loss, ce, mse, validation_loss, validation_ce, validation_mse, accuracy,
                                         precision, recall]]
        info_str = ','.join(info) + '\n'

        with open(Config.TRAINING_RESULTS_FILE, 'a') as f:
            f.write(info_str)

        torch.save(model.state_dict(), f'{Config.CHECKPOINT_DIR}/{checkpoint_name}')
