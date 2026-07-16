import torch
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self, dropout_probability=0.2):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)
    

# if __name__ == "__main__":
#     model = MNISTClassifier()
#     
#     sample_batch = torch.randn(64, 1, 28, 28)  # Example batch of 64 MNIST images
#     output = model(sample_batch)
#     
#     print("Output batch shape:", output.shape)
#     print("Input batch shape:", sample_batch.shape)
