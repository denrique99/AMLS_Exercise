import torch.nn as nn

# Define the CNN model for ECG classification
# The model architecture is a simple CNN with two convolutional layers followed by two fully connected layers.
# inherit from nn.Module
class ECGCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ECGCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
           nn.Linear(35840, 64),  # ggf. anpassen!
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x