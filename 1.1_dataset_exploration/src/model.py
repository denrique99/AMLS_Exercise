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
        self.fc = None  # to be defined after seeing input

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)

        if self.fc is None:
            # Define fully connected layer dynamically based on input
            self.fc = nn.Sequential(
                nn.Linear(x.shape[1], 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 4)
            )
            self.fc.to(x.device)

        x = self.fc(x)
        return x










# import torch.nn as nn

# # Define the CNN model for ECG classification
# # The model architecture is a simple CNN with two convolutional layers followed by two fully connected layers.
# # inherit from nn.Module
# class ECGCNN(nn.Module):
#     def __init__(self, num_classes=4):
#         super(ECGCNN, self).__init__()
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Dropout(0.1),

#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Dropout(0.25)
#         )
#         self.flatten = nn.Flatten()
#         self.fc = nn.Sequential(
#             nn.Linear(35840, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):
#         x = self.conv_block(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x

