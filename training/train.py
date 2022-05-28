import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config
from model.loss import DiscretizedMixtureLogisticLoss
from model.modules.upsampling import ConvInUpsampleNetwork
from model.wavenet import WaveNet
from training.dataset import get_dataset

if __name__ == '__main__':
    print(f'Running training on {torch.cuda.get_device_name(torch.cuda.current_device())}')

    upsample_net = ConvInUpsampleNetwork(
        upsample_scales=Config.upsample_scales,
        cin_channels=Config.num_mels
    )
    model = WaveNet(
        out_channels=Config.out_channels,
        layers=Config.layers,
        stacks=Config.stacks,
        residual_channels=Config.residual_channels,
        gate_channels=Config.gate_channels,
        skip_channels=Config.skip_channels,
        local_conditioning_channels=Config.num_mels,
        dropout_probability=Config.dropout_probability,
        kernel_size=Config.kernel_size,
        upsample_net=upsample_net
    ).cuda()
    model.train()

    Config.in_seq_len = model.receptive_field
    Config.out_seq_len = model.receptive_field

    error = DiscretizedMixtureLogisticLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)

    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    for epoch in range(Config.epochs):
        epoch_losses = []
        for batch_num, (X, Cond, Y) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            X = X.cuda()
            Cond = Cond.cuda()
            Y = Y.cuda()

            Y_pred = model(X, Cond)

            input_lengths = torch.LongTensor([len(y) for y in Y]).cuda()
            loss = error(
                input=Y_pred,
                target=Y,
                lengths=input_lengths
            )
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            print(f'Epoch: {epoch} Batch: {batch_num}/{len(dataloader)} Loss: {loss.item()}')
            if batch_num % 20 == 0:
                checkpoint_name = f'wavenet_epoch{epoch}_batchNum{batch_num}_withUpsampling_withMelSpec'
                torch.save(model.state_dict(), f'..\\saved_models\\{checkpoint_name}')

        checkpoint_name = f'wavenet_epoch{epoch}_withUpsampling_withMelSpec'
        with open('losses.txt', 'a') as f:
            f.write(f'Model: {checkpoint_name} - LOSS: {torch.mean(torch.Tensor(epoch_losses))}\n')
        torch.save(model.state_dict(), f'..\\saved_models\\{checkpoint_name}')
