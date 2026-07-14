import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

from model import MNISTClassifier

from pathlib import Path

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transform,
)

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)

model = MNISTClassifier().to(device)

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=0.001
)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    
    total_loss = 0
    correct_predictions = 0
    total_examples = 0
    
    for images, labels in train_loader:
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = torch.argmax(logits, dim=1)
        
        correct_predictions += (
            (predictions == labels).sum().item()
        )
        total_examples += labels.size(0)
        
    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples

    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}, "
        f"Accuracy: {accuracy:.4f}"
    )
    
model.eval()

test_loss = 0
correct_test_predictions = 0
total_test_examples = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        loss = loss_function(logits, labels)
        
        test_loss += loss.item() * images.size(0)
        test_predictions = torch.argmax(logits, dim=1)
        
        correct_test_predictions += (test_predictions == labels).sum().item()
        total_test_examples += labels.size(0)
        
average_loss = test_loss / total_test_examples
accuracy = correct_test_predictions / total_test_examples

print(
    f"Test Loss: {average_loss:.4f}, "
    f"Test Accuracy: {accuracy:.4f}"
) 
    
models_directory = Path("../models")
models_directory.mkdir(parents=True, exist_ok=True)

model_path = models_directory / "mnist_mlp.pth"

torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


    
    
        