import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config
from midi.dataset import MidiDataset
from model.LSTMidi import LSTMidi
from training.loss import custom_two_part_loss
from training.lrschedule import step_learning_rate_decay

if __name__ == '__main__':
    print(f'Running training on {torch.cuda.get_device_name(torch.cuda.current_device())}')

    model = LSTMidi().cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    dataset = MidiDataset.load(Config.TRAINING_DATASET)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    step = 0
    current_lr = Config.LEARNING_RATE

    for epoch in range(Config.EPOCHS):
        epoch_losses = []
        progress_bar = tqdm(enumerate(dataloader),
                            unit='batches',
                            total=len(dataloader),
                            desc=f'Epoch {epoch}')
        for batch_num, (X, pitch, extra) in progress_bar:
            step += 1

            if Config.DYNAMIC_LR:
                current_lr = step_learning_rate_decay(
                    Config.LEARNING_RATE, step, Config.ANNEAL_RATE, Config.ANNEAL_INTERVAL)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            optimizer.zero_grad()

            X = X.cuda()
            pitch = pitch.cuda()
            extra = extra.cuda()

            pitch_pred, extra_pred = model(X)

            cross_entropy, mse, loss = custom_two_part_loss(pitch, extra, pitch_pred, extra_pred)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            progress_bar.set_postfix_str(f'Loss: {loss.item()} (CE: {cross_entropy.item()}, MSE: {mse.item()}')
            if batch_num % Config.SAVE_INTERVAL_BATCHES == 0:
                checkpoint_name = f'{Config.MODEL_NAME}_Epoch{epoch}_BatchNum{batch_num}_Loss{loss.item()}'
                torch.save(model.state_dict(),
                           f'..\\saved_models\\{Config.MODEL_NAME}_{Config.SESSION}\\{checkpoint_name}')

        checkpoint_name = f'{Config.MODEL_NAME}_{Config.SESSION}_Epoch{epoch}'
        with open('losses.txt', 'a') as f:
            f.write(f'Model: {checkpoint_name} - LOSS: {torch.mean(torch.Tensor(epoch_losses))}\n')

        torch.save(model.state_dict(), f'..\\saved_models\\{Config.MODEL_NAME}_{Config.SESSION}\\{checkpoint_name}')
