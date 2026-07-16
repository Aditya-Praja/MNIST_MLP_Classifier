import torch
from torch.utils.data import DataLoader, random_split
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

training_size = 50000
validation_size = 10000

training_dataset, validation_dataset = random_split(
    train_dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(42),
)


# --------------------------------------------------
# 2. Create DataLoaders
# --------------------------------------------------

batch_size = 64

train_loader = DataLoader(
    training_dataset,
    batch_size=batch_size,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
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

validation_losses = []
validation_accuracies = []

best_validation_loss = float("inf")
best_epoch = 0


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

    total_validation_loss = 0.0
    correct_validation_predictions = 0
    total_validation_examples = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = loss_function(
                logits,
                labels,
            )

            total_validation_loss += (
                loss.item() * images.size(0)
            )

            validation_predictions = logits.argmax(
                dim=1
            )

            correct_validation_predictions += (
                validation_predictions == labels
            ).sum().item()

            total_validation_examples += labels.size(0)

    average_validation_loss = (
        total_validation_loss
        / total_validation_examples
    )

    validation_accuracy = (
        correct_validation_predictions
        / total_validation_examples
    )

    validation_losses.append(
        average_validation_loss
    )

    validation_accuracies.append(
        validation_accuracy
    )
    
    if average_validation_loss < best_validation_loss:
        best_validation_loss = average_validation_loss
        best_epoch = epoch
        
        models_directory = Path("../models")
        models_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        
        best_model_path = (
            models_directory / "best_mnist_model.pth"
        )
        
        torch.save(
            model.state_dict(),
            best_model_path,
        )
        
        print(
            f"Saved best model to epoch {best_epoch}"
            f"with validation loss "
            f"{best_validation_loss:.4f}"
        )

    # ==========================
    # Print this epoch's results
    # ==========================

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | "
        f"Train Loss: {average_training_loss:.4f} | "
        f"Train Accuracy: {training_accuracy:.4f} | "
        f"Validation Loss: {average_validation_loss:.4f} | "
        f"Validation Accuracy: {validation_accuracy:.4f}"
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
    validation_accuracies,
    marker="o",
    label="Validation Accuracy",
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
    validation_losses,
    marker="o",
    label="Validation Loss",
)

plt.title("Loss Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.xticks(epochs)
plt.legend()
plt.tight_layout()
plt.show()

plots_directory = Path("../plots")
plots_directory.mkdir(
    parents=True,
    exist_ok=True,
)

accuracy_plot_path = plots_directory / "accuracy_across_epochs.png"
loss_plot_path = plots_directory / "loss_across_epochs.png"

plt.savefig(accuracy_plot_path)
plt.savefig(loss_plot_path)


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