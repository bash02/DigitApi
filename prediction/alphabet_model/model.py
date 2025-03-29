import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# === CONFIGURABLE PARAMETERS === #
DATA_DIR = './alphabet'  # Path to extracted alphabet dataset
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'alphabet_model.pth')
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# Ensure dataset exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset folder '{DATA_DIR}' not found! Extract alphabets first.")

# === DEFINE CNN MODEL === #
class AlphabetCNN(nn.Module):
    def __init__(self):
        super(AlphabetCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 7 * 7, 26)  # Output 26 classes (A-Z)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

# === LOAD DATASET === #
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# === DEVICE SETUP === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphabetCNN().to(device)  # Move model to GPU if available

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
            images, labels = images.to(device), labels.to(device)  # ✅ Corrected `.to(device)`

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



















# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import os

# # 1. Define dataset directory
# data_dir = './alphabet'  # Folder should contain subfolders A-Z

# # Ensure dataset exists
# if not os.path.exists(data_dir) or not os.listdir(data_dir):
#     raise FileNotFoundError(f"Dataset directory {data_dir} is empty or missing!")

# # 2. Define the model
# class AlphabetCNN(nn.Module):
#     def __init__(self):
#         super(AlphabetCNN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
#         )
#         self.fc = nn.Linear(64 * 7 * 7, 26)  # Output 26 classes (A-Z)

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)

# # 3. Define model path
# model_path = os.path.join(os.path.dirname(__file__), 'alphabet_model.pth')

# # 4. Load dataset with preprocessing
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# train_data = datasets.ImageFolder(root=data_dir, transform=transform)
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# # 5. Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AlphabetCNN().to(device)

# # 6. Check if the trained model exists
# if os.path.exists(model_path):
#     print("Loading existing trained model...")
#     model.load_state_dict(torch.load(model_path, map_location=device))
# else:
#     print("Training new alphabet model...")
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Train for 5 epochs
#     model.train()
#     for epoch in range(5):
#         total_loss = 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
        
#         print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

#     # Save model
#     torch.save(model.state_dict(), model_path)
#     print(f"Alphabet model trained and saved at: {model_path}")





