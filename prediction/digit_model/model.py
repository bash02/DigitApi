import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# === CONFIGURABLE PARAMETERS === #
DATA_DIR = './digit'  # Path to extracted digit dataset
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'digit_model.pth')
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# Ensure dataset exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset folder '{DATA_DIR}' not found! Extract digits first.")

# === DEFINE CNN MODEL === #
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)  # 10 classes for digits (0-9)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# === LOAD DATASET === #
transform = transforms.Compose([
    transforms.Grayscale(),  # Ensure images are grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# === DEVICE SETUP === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitCNN().to(device)

# === LOAD OR TRAIN MODEL === #
if os.path.exists(MODEL_PATH):
    print("Loading existing trained model...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    CONTINUE_TRAINING = True  # Allow further training
else:
    print("No pre-trained model found. Training a new model...")
    CONTINUE_TRAINING = True

# === TRAINING FUNCTION === #
def train_model(model, train_loader, device, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")

# === TRAIN ONLY IF CONTINUE_TRAINING IS TRUE === #
if CONTINUE_TRAINING:
    print("Continuing training on existing model...")
train_model(model, train_loader, device, EPOCHS, LEARNING_RATE)
