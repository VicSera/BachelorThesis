import torch
from torch.utils.data import DataLoader

from core.config import Config
from midi.dataset import MidiDataset
from model.LSTMidi import LSTMidi
from training.loss import custom_two_part_loss
from training.lrschedule import step_learning_rate_decay


def progress_fn(step, total_steps):
    print(f"Step {step}/{total_steps}")


if __name__ == '__main__':
    print(f'Running training on {torch.cuda.get_device_name(torch.cuda.current_device())}')

    model = LSTMidi().cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    dataset = MidiDataset.load(Config.TRAINING_DATASET)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    step = 0

    for epoch in range(Config.EPOCHS):
        epoch_losses = []
        for batch_num, (X, Y) in enumerate(dataloader):
            step += 1

            current_lr = step_learning_rate_decay(
                Config.LEARNING_RATE, step, Config.ANNEAL_RATE, Config.ANNEAL_INTERVAL)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            X = X.cuda()

            Y_pred = model(X)

            loss = custom_two_part_loss(Y, Y_pred)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            print(f'Epoch: {epoch} Batch: {batch_num}/{len(dataloader)} Loss: {loss.item()} LR: {current_lr}')
            if batch_num % 20 == 0:
                checkpoint_name = f'LSTMidi_Epoch{epoch}_BatchNum{batch_num}'
                # torch.save(model.state_dict(), f'..\\saved_models\\LSTMidi\\{checkpoint_name}')

        checkpoint_name = f'LSTMidi_Epoch{epoch}'
        with open('losses.txt', 'a') as f:
            f.write(f'Model: {checkpoint_name} - LOSS: {torch.mean(torch.Tensor(epoch_losses))}\n')
