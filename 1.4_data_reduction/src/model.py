import torch
import torch.nn as nn

class ECGCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ECGCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.1),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25)
        )

        self.flatten = nn.Flatten()

        # Define a placeholder Linear layer; it will be re-initialized later
        self.fc_template = nn.Sequential(
            nn.Linear(1, 64),  # placeholder input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        self.fc = None

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)

        # Dynamically define self.fc with correct input size
        if self.fc is None:
            self.fc = nn.Sequential(
                nn.Linear(x.shape[1], 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 4)
            )
            self.fc.to(x.device)

        return self.fc(x)






