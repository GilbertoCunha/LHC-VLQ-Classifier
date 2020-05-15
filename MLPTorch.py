import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np


# Selecionar o gpu se possível
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = 'cpu'


# Criar o modelo
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=36*36).to(device)
        self.fc2 = nn.Linear(in_features=36*36, out_features=40*40).to(device)
        self.fc3 = nn.Linear(in_features=40*40, out_features=num_classes).to(device)
        self.dropout = nn.Dropout(0.2)

    def forward(self, t):
        t = self.dropout(F.relu(self.fc1(t.reshape(-1, 28*28))))
        t = F.relu(self.fc2(t))
        t = F.softmax(self.fc3(t), dim=-1)
        return t


# Load MNIST Datasets
dataset = torchvision.datasets.MNIST(
    root='',
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ]),
    train=True
)
test_set = torchvision.datasets.MNIST(
    root='',
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ]),
    train=False
)

# Generate loaders
train_size = 0.9*len(dataset)
splitting = [int(train_size), len(dataset) - int(train_size)]
train_set, valid_set = torch.utils.data.random_split(dataset, splitting)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)

# Parametros de ajuste do modelo
model = MLP(28*28, 10).to(device)
optimizer = torch.optim.Adam(model.parameters())
epochs = 10
best_valid_loss = float("inf")

# Ciclo de treino
for epoch in range(epochs):

    # Training
    train_acc, train_loss = 0, 0
    for images, labels in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1} Train"):
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        loss = F.cross_entropy(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += predictions.argmax(dim=-1).eq(labels).sum().item()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader.dataset)

    # Validation
    valid_loss, valid_acc = 0, 0
    for images, labels in tqdm(valid_loader, total=len(valid_loader), desc=f"Epoch {epoch+1} Valid"):
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)

        loss = F.cross_entropy(predictions, labels)

        optimizer.zero_grad()
        loss.backward()

        valid_acc += predictions.argmax(dim=-1).eq(labels).sum().item()
        valid_loss += loss.item()
    valid_loss = valid_loss / len(valid_loader)
    valid_acc = valid_acc / len(valid_loader.dataset)

    # Save model
    if valid_loss < best_valid_loss:
        best_valid_ac = valid_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'valid_loss': valid_loss,
            'valid_acc': valid_acc
        }, "Models/torch_model.tar")

    tqdm.write(f"Epoch {epoch+1} loss: {train_loss} | Epoch {epoch+1} acc: {train_acc}")
