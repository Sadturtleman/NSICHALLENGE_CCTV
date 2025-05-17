import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from model import CSRNet
from dataset import CrowdDataset

def train():
    image_dir = r'archive\ShanghaiTech\part_B\train_data\images'
    density_dir = r'archive\ShanghaiTech\part_B\train_data\density-map'

    dataset = CrowdDataset(image_dir, density_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = CSRNet().cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        for i, (images, targets) in enumerate(dataloader):
            images, targets = images.cuda(), targets.cuda()

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch}] Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f'csrnet_epoch{epoch}.pth')

if __name__ == '__main__':
    train()
