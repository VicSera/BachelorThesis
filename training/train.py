import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from core.config import Config
from model.loss import DiscretizedMixtureLogisticLoss
from model.wavenet_2 import WaveNetModel
from training.dataset import get_dataset, DrumDataset
from training.lrschedule import step_learning_rate_decay


def progress_fn(step, total_steps):
    print(f"Step {step}/{total_steps}")


if __name__ == '__main__':
    print(f'Running training on {torch.cuda.get_device_name(torch.cuda.current_device())}')

    model = WaveNetModel().cuda()
    model.train()

    error = DiscretizedMixtureLogisticLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)

    dataset = DrumDataset.load(Config.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    step = 0

    for epoch in range(Config.epochs):
        epoch_losses = []
        for batch_num, (X, Cond, Y) in enumerate(dataloader):
            step += 1

            current_lr = step_learning_rate_decay(
                Config.learning_rate, step, Config.anneal_rate, Config.anneal_interval)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            optimizer.zero_grad()

            X = X.cuda()
            Cond = Cond.cuda()
            Y = Y.type('torch.LongTensor').cuda().squeeze(2)

            Y_pred = model(X, Cond).transpose(1, 2).contiguous()

            loss = F.cross_entropy(
                input=Y_pred,
                target=Y,
            )
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            print(f'Epoch: {epoch} Batch: {batch_num}/{len(dataloader)} Loss: {loss.item()} LR: {current_lr}')
            if batch_num % 20 == 0:
                checkpoint_name = f'wavenet_epoch{epoch}_batchNum{batch_num}'
                torch.save(model.state_dict(), f'..\\saved_models\\wavenet2_ed3_withConv\\{checkpoint_name}')

        checkpoint_name = f'wavenet_epoch{epoch}_withUpsampling_withMelSpec'
        with open('losses.txt', 'a') as f:
            f.write(f'Model: {checkpoint_name} - LOSS: {torch.mean(torch.Tensor(epoch_losses))}\n')
        torch.save(model.state_dict(), f'..\\saved_models\\{checkpoint_name}')
