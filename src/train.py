import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pathlib import Path
import matplotlib.pyplot as plt

from model import MNISTClassifier


# --------------------------------------------------
# 1. Load the MNIST datasets
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Create DataLoaders
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Select the device
# --------------------------------------------------

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# 4. Create the model, loss function, and optimizer
# --------------------------------------------------

model = MNISTClassifier().to(device)

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)


# --------------------------------------------------
# 5. Create lists for tracking history
# --------------------------------------------------

training_losses = []
training_accuracies = []

test_losses = []
test_accuracies = []


# --------------------------------------------------
# 6. Train and evaluate after every epoch
# --------------------------------------------------

num_epochs = 10

for epoch in range(num_epochs):

    # ==========================
    # Training phase
    # ==========================

    model.train()

    total_training_loss = 0.0
    correct_training_predictions = 0
    total_training_examples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients left over from the previous batch.
        optimizer.zero_grad()

        # Forward pass.
        logits = model(images)

        # Calculate the batch loss.
        loss = loss_function(logits, labels)

        # Calculate gradients.
        loss.backward()

        # Update the model parameters.
        optimizer.step()

        # Accumulate the total loss for this epoch.
        total_training_loss += (
            loss.item() * images.size(0)
        )

        # Select the class with the highest logit.
        training_predictions = logits.argmax(dim=1)

        correct_training_predictions += (
            training_predictions == labels
        ).sum().item()

        total_training_examples += labels.size(0)

    average_training_loss = (
        total_training_loss
        / total_training_examples
    )

    training_accuracy = (
        correct_training_predictions
        / total_training_examples
    )

    training_losses.append(
        average_training_loss
    )

    training_accuracies.append(
        training_accuracy
    )


    # ==========================
    # Evaluation phase
    # ==========================

    model.eval()

    total_test_loss = 0.0
    correct_test_predictions = 0
    total_test_examples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = loss_function(
                logits,
                labels,
            )

            total_test_loss += (
                loss.item() * images.size(0)
            )

            test_predictions = logits.argmax(
                dim=1
            )

            correct_test_predictions += (
                test_predictions == labels
            ).sum().item()

            total_test_examples += labels.size(0)

    average_test_loss = (
        total_test_loss
        / total_test_examples
    )

    test_accuracy = (
        correct_test_predictions
        / total_test_examples
    )

    test_losses.append(
        average_test_loss
    )

    test_accuracies.append(
        test_accuracy
    )


    # ==========================
    # Print this epoch's results
    # ==========================

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | "
        f"Train Loss: {average_training_loss:.4f} | "
        f"Train Accuracy: {training_accuracy:.4f} | "
        f"Test Loss: {average_test_loss:.4f} | "
        f"Test Accuracy: {test_accuracy:.4f}"
    )
    
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    training_accuracies,
    marker="o",
    label="Training Accuracy",
)

plt.plot(
    epochs,
    test_accuracies,
    marker="o",
    label="Test Accuracy",
)

plt.title("Accuracy Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs)
plt.legend()
plt.tight_layout()
plt.show()

# Plot the loss curves separately
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    training_losses,
    marker="o",
    label="Training Loss",
)

plt.plot(
    epochs,
    test_losses,
    marker="o",
    label="Test Loss",
)

plt.title("Loss Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.xticks(epochs)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 7. Save the trained model
# --------------------------------------------------

models_directory = Path("../models")

models_directory.mkdir(
    parents=True,
    exist_ok=True,
)

model_path = (
    models_directory / "mnist_mlp.pth"
)

torch.save(
    model.state_dict(),
    model_path,
)

print(f"Model saved to {model_path}")