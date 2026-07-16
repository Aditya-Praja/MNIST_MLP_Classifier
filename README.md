MNIST MLP Classifier

A handwritten-digit classifier built with PyTorch. The project trains a multilayer perceptron to classify MNIST images into digits from 0 through 9.

The project includes training, validation, early stopping, checkpointing, evaluation, error analysis, training plots, and prediction on custom handwritten images.

Project Overview

Each MNIST image is a grayscale image with dimensions:

1 × 28 × 28

The image is flattened into 784 pixel features and passed through a fully connected neural network.

Input image: [1, 28, 28]
        ↓
Flatten: 784 features
        ↓
Linear: 784 → 128
        ↓
ReLU
        ↓
Dropout
        ↓
Linear: 128 → 64
        ↓
ReLU
        ↓
Dropout
        ↓
Linear: 64 → 10 logits
        ↓
Predicted digit: 0–9

Features

* PyTorch model implementation
* MNIST loading through torchvision
* Training, validation, and test splits
* Mini-batch training with DataLoader
* Adam optimization
* Cross-entropy loss
* Dropout regularization
* Early stopping
* Best-model checkpointing
* Accuracy and loss plots
* Confusion-matrix analysis
* Prediction confidence scores
* Custom handwritten-image prediction
* Command-line configuration

Project Structure

MNIST_MLP_Classifier/
├── data/
│   └── Downloaded MNIST data
├── models/
│   └── Saved model checkpoints
├── plots/
│   ├── accuracy.png
│   └── loss_curve.png
├── sample_images/
│   └── Custom handwritten-digit images
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── .gitignore
├── README.md
└── requirements.txt

The data/, models/, and .venv/ directories are ignored by Git because they contain generated or environment-specific files.

Installation

Clone the repository:

git clone https://github.com/Aditya-Praja/MNIST_MLP_Classifier.git
cd MNIST_MLP_Classifier

Create a virtual environment:

python3 -m venv .venv

Activate it on macOS or Linux:

source .venv/bin/activate

Install the dependencies:

python3 -m pip install -r requirements.txt

Training

Run training from the project root:

python3 src/train.py

Example with custom hyperparameters:

python3 src/train.py \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --dropout 0.2 \
    --patience 3

View all available options:

python3 src/train.py --help

Training configuration

The default configuration uses:

Optimizer: Adam
Loss function: CrossEntropyLoss
Batch size: 64
Learning rate: 0.001
Dropout probability: 0.2
Maximum epochs: 20
Early-stopping patience: 3

During training, the original 60,000-image MNIST training set is divided into:

Training examples:   50,000
Validation examples: 10,000

The official MNIST test set contains another 10,000 images and is evaluated only after training finishes.

Model Selection

The model is evaluated on the validation set after every epoch.

When validation loss improves, the project saves a checkpoint containing:

* Model parameters
* Optimizer state
* Epoch number
* Best validation loss
* Training history
* Hyperparameter configuration

Training stops early when validation loss fails to improve for the configured number of consecutive epochs.

The best saved checkpoint—not necessarily the final epoch—is used for final test evaluation.

Evaluation

Run:

python3 src/evaluate.py

The evaluation script reports:

* Test loss
* Test accuracy
* Prediction probabilities
* Incorrect predictions
* Confusion matrix
* Most common misclassification

Results

Example results from the MLP model:

Final test accuracy: approximately 97–98%
Most common mistake: actual 4 predicted as 9

Replace this section with the exact output from the final best checkpoint:

Best epoch: [INSERT]
Best validation loss: [INSERT]
Final test loss: [INSERT]
Final test accuracy: [INSERT]

Training Curves

Accuracy

Loss

The curves help show whether the model is continuing to generalize or beginning to overfit.

A typical overfitting pattern is:

Training loss continues decreasing
Validation loss begins increasing

Early stopping and best-checkpoint saving reduce the effect of this behavior.

Error Analysis

The confusion matrix records:

Rows: actual digit
Columns: predicted digit

For example:

confusion_matrix[4, 9]

represents the number of actual handwritten 4s classified as 9s.

The model commonly confuses digits that have visually similar strokes, loops, or shapes. Inspecting these mistakes provides more information than accuracy alone.

Predicting a Custom Image

Place a handwritten-digit image in sample_images/, then run:

python3 src/predict.py sample_images/digit_7.png

Example output:

Predicted digit: 7
Confidence: 98.71%

The custom-image preprocessing pipeline:

1. Converts the image to grayscale.
2. Detects whether the background should be inverted.
3. Crops surrounding whitespace.
4. Preserves the digit’s aspect ratio.
5. Resizes the digit to fit within a 20 × 20 region.
6. Centers it on a 28 × 28 black canvas.
7. Converts it to a PyTorch tensor.
8. Adds a batch dimension.

Custom images may still perform worse than MNIST images because handwriting style, stroke thickness, positioning, and image quality may differ from the training distribution.

Key Concepts Demonstrated

This project demonstrates:

* Image tensors
* Batch dimensions
* Multiclass classification
* Logits and softmax probabilities
* Cross-entropy loss
* Backpropagation
* Gradient-based optimization
* Training and evaluation modes
* Dropout regularization
* Data splitting
* Early stopping
* Model checkpointing
* Generalization and overfitting
* Confusion matrices
* Error analysis

Limitations

The model is a multilayer perceptron, so it flattens each image before processing it.

Flattening removes explicit spatial relationships between neighboring pixels. A convolutional neural network would better capture local image patterns such as:

* Edges
* Curves
* Corners
* Loops
* Stroke combinations

A future extension could compare this MLP against a CNN on the same dataset.

Future Improvements

* Add a convolutional neural network
* Compare multiple dropout values
* Track experiments in a CSV or experiment-management tool
* Add automated tests
* Add normalization using MNIST mean and standard deviation
* Improve custom-image centering using the digit’s center of mass
* Export the trained model for deployment

Author

Aditya Prajapati