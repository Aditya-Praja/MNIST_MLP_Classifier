import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from pathlib import Path
import matplotlib.pyplot as plt

from model import MNISTClassifier

def train_one_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device,
):
    model.train()
    
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0
    
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = loss_function(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        predictions = logits.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_examples += labels.size(0)
        
    average_loss = total_loss/total_examples
    accuracy = correct_predictions / total_examples
    
    return average_loss, accuracy
        
        
def evaluate(
    model,
    data_loader,
    loss_function,
    device,
):
    model.eval()
    
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_function(logits, labels)

            total_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples

    return average_loss, accuracy

def save_training_plot(
    training_losses,
    validation_losses,
    training_accuracies,
    validation_accuracies,
    plots_directory,
):
    
    plots_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    completed_epochs = len(training_losses)
    epochs = range(1, completed_epochs + 1)
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(
        epochs,
        training_accuracies,
        marker="o",
        label="Training Accuracy"
    )
    
    plt.plot(
        epochs,
        validation_accuracies,
        marker="o",
        label="Validation Accuracy"
    )
    
    plt.title("Accuracy Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.legend()
    plt.tight_layout()
    
    accuracy_path = plots_directory / "accuracy.png"
    plt.savefig(accuracy_path)
    plt.close()
    
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
    
    loss_path = plots_directory / "loss_curve.png"
    plt.savefig(loss_path)
    plt.close()
    
    print(f"Saved accuracy plot to {accuracy_path}")
    print(f"Saved loss plot to {loss_path}")



def main():
    
    batch_size = 64
    learning_rate = 0.001
    dropout_probability = 0.2
    num_epochs = 20
    patience = 3
    
    data_directory = Path("../data")
    models_directory = Path("../models")
    plots_directory = Path("../plots")
    models_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    plots_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    best_model_path = (
        models_directory / "best_mnist_mlp.pth"
    )
    
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_directory,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_directory,
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

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    model = MNISTClassifier(dropout_probability=dropout_probability).to(device)

    loss_function = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    training_losses = []
    training_accuracies = []

    validation_losses = []
    validation_accuracies = []

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        
        average_training_loss, training_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                loss_function,
                optimizer,
                device,
            )
        )
        
        average_validation_loss, validation_accuracy = (
            evaluate(
                model,
                validation_loader,
                loss_function,
                device,
            )
        )

        training_losses.append(
            average_training_loss
        )

        training_accuracies.append(
            training_accuracy
        )

        validation_losses.append(
            average_validation_loss
        )

        validation_accuracies.append(
            validation_accuracy
        )
        
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {average_training_loss:.4f} | "
            f"Train Accuracy: {training_accuracy:.4f} | "
            f"Validation Loss: {average_validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )
        
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            torch.save(
                model.state_dict(),
                best_model_path,
            )
            
            print(
                f"Saved best model to epoch {best_epoch}"
                f" with validation loss "
                f"{best_validation_loss:.4f}"
            )
        
        else:
            epochs_without_improvement += 1
            
            print("Validation loss did not improve. "
                f"Patience: {epochs_without_improvement}/{patience}"
            )
        
        if epochs_without_improvement >= patience:
            print(f"\nEarly stop triggered after epoch {epoch + 1}")
            break

    save_training_plot(
        training_losses,
        validation_losses,
        training_accuracies,
        validation_accuracies,
        plots_directory,
    )
    
    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
            weights_only=True,
        )
    )

    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
    )

    print(
        f"\nBest epoch: {best_epoch} | "
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Final Test Loss: {test_loss:.4f} | "
        f"Final Test Accuracy: {test_accuracy:.4f}"
    )
    
if __name__ == "__main__":
    main()