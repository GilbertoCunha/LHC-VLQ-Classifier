import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = 'cpu'


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=36*36).to(device)
        self.fc2 = nn.Linear(in_features=36*36, out_features=40*40).to(device)
        self.fc3 = nn.Linear(in_features=40*40, out_features=num_classes).to(device)

    def forward(self, t):
        t = F.relu(self.fc1(t.reshape(-1, 28*28)))
        t = F.relu(self.fc2(t))
        t = F.softmax(self.fc3(t), dim=-1)
        return t


dataset = torchvision.datasets.MNIST(
    root='',
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ]),
    train=True
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Parameters
model = MLP(28*28, 10).to(device)
optimizer = torch.optim.Adam(model.parameters())
epochs = 20

for epoch in range(epochs):
    train_acc, train_loss = 0, 0

    for image, label in tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}"):
        image, label = image.to(device), label.to(device)
        prediction = model(image)
        loss = F.cross_entropy(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += prediction.argmax(dim=-1).eq(label).sum().item()
        train_loss += loss.item()

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader.dataset)

    print(f"Epoch {epoch+1} loss: {train_loss} | Epoch {epoch+1} acc: {train_acc}")
