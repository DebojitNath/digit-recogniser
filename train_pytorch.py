from torch.optim import optimizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim


# ==========================================
# STEP 1: LOAD THE DATA
# ==========================================


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

print("Downloading training data...")
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

print("Downloading testing data...")
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
print(f"Data loaded! Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")


# ==========================================
# STEP 2: DEFINE THE MODEL ARCHITECTURE
# ==========================================


class DigitalCNN(nn.Module):
    def __init__(self):
        super(DigitalCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

model = DigitalCNN()
print("Model initialized...")


# ==========================================
# STEP 3: TRAIN AND SAVE THE MODEL
# ==========================================


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 2

print("\nStarting Training...")
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # Zero the gradients (resetting the optimizer for this batch)
        optimizer.zero_grad()
        # Forward pass (make a prediction)
        outputs = model(inputs)
        # Calculate how wrong the prediction was
        loss = criterion(outputs, labels)
        # Backward pass (calculate how to adjust the weights)
        loss.backward()
        # Optimize (actually adjust the weights)
        optimizer.step()
        # Print statistics every 300 batches
        running_loss += loss.item()
        if i % 300 == 299:
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 300:.3f}")
            running_loss = 0.0
print("Finished Training!")
# Save the model
MODEL_PATH = os.environ.get('MODEL_PATH', 'mnist_cnn.pth')
torch.save(model.state_dict(), MODEL_PATH)
print(f"Training complete! Model saved to {MODEL_PATH}")