import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define data directories
train_dir = 'train_split'
test_dir = 'test_split'
valid_dir = 'val_split'

# Data augmentation transforms
transform = transforms.Compose([
    transforms.RandomRotation(90),  # Rotate images by 90 degrees
    transforms.Resize((224, 224)),  # Resize images to the input size of ResNet-34
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values
])

# Load datasets
train_dataset = ImageFolder(train_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)
valid_dataset = ImageFolder(valid_dir, transform=transform)

# Create data loaders
batch_sizes = [16, 32, 64]
learning_rates = [0.01, 0.05, 0.1]

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

        # Load pretrained ResNet-34
        model = torch.hub.load('pytorch/vision', 'resnet34', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification (2 classes)

        # Set hyperparameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in valid_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Accuracy: {accuracy:.2f}%")

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Test Accuracy: {test_accuracy:.2f}%")
