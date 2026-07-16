import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import MNISTClassifier


transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
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

model = MNISTClassifier().to(device)

model.load_state_dict(
    torch.load(
        "../models/best_mnist_model.pth",
        map_location=device,
        weights_only=True,
    )
)

model.eval()

loss_function = torch.nn.CrossEntropyLoss()

confusion_matrix = torch.zeros(
    10,
    10,
    dtype=torch.int64,
)

four_as_nine_images = []
four_as_nine_confidences = []

total_loss = 0.0
correct_predictions = 0
total_examples = 0


# Evaluate the entire test set
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_function(logits, labels)

        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        confidence_scores = probabilities.max(dim=1).values

        # CrossEntropyLoss gives the mean loss for this batch.
        # Multiply by batch size to recover the batch's total loss.
        total_loss += loss.item() * images.size(0)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_examples += labels.size(0)

        # Locate actual 4s that were predicted as 9s.
        mistake_mask = (labels == 4) & (predictions == 9)
        mistake_indices = mistake_mask.nonzero(as_tuple=True)[0]

        for index in mistake_indices:
            four_as_nine_images.append(
                images[index].cpu()
            )

            four_as_nine_confidences.append(
                confidence_scores[index].item()
            )

        # Update the CPU confusion matrix.
        for actual, predicted in zip(labels, predictions):
            confusion_matrix[
                actual.item(),
                predicted.item(),
            ] += 1


average_loss = total_loss / total_examples
accuracy = correct_predictions / total_examples

print(f"Average loss: {average_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")


# Inspect the first test batch
images, labels = next(iter(test_loader))

images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    logits = model(images)
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    confidence_scores = probabilities.max(dim=1).values

for i in range(10):
    print(
        f"Image {i + 1}: "
        f"Predicted label = {predictions[i].item()}, "
        f"Confidence = {confidence_scores[i].item():.4f}, "
        f"Ground truth = {labels[i].item()}"
    )


# Inspect incorrect predictions from the first batch
incorrect_indices = (
    predictions != labels
).nonzero(as_tuple=True)[0]

incorrect_labels = labels[incorrect_indices]
incorrect_predictions = predictions[incorrect_indices]
incorrect_confidences = confidence_scores[incorrect_indices]

print(
    "Incorrect predictions:",
    incorrect_predictions.cpu(),
)

print(
    "Ground truth labels for incorrect predictions:",
    incorrect_labels.cpu(),
)

print(
    "Confidence scores for incorrect predictions:",
    incorrect_confidences.cpu(),
)


# Display the confusion matrix
plt.figure(figsize=(8, 8))
plt.imshow(confusion_matrix.numpy())

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(range(10))
plt.yticks(range(10))
plt.colorbar()

for actual in range(10):
    for predicted in range(10):
        count = confusion_matrix[
            actual,
            predicted,
        ].item()

        plt.text(
            predicted,
            actual,
            str(count),
            ha="center",
            va="center",
        )

plt.tight_layout()
plt.show()


# Find the most common misclassification
mistakes_only = confusion_matrix.clone()
mistakes_only.fill_diagonal_(0)

largest_mistake_index = mistakes_only.argmax().item()

actual_digit = largest_mistake_index // 10
predicted_digit = largest_mistake_index % 10

mistake_count = mistakes_only[
    actual_digit,
    predicted_digit,
].item()

print(
    f"Most common mistake: actual {actual_digit} "
    f"predicted as {predicted_digit} "
    f"{mistake_count} times"
)


# Display all actual 4s predicted as 9s
print(
    f"Found {len(four_as_nine_images)} "
    "actual 4s predicted as 9s."
)

for image, confidence in zip(
    four_as_nine_images,
    four_as_nine_confidences,
):
    image = image.squeeze()

    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap="gray")

    plt.title(
        "Actual: 4 | Predicted: 9\n"
        f"Confidence: {confidence:.4f}"
    )

    plt.axis("off")
    plt.show()