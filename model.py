import torch
import torch.nn as nn


class catdogcnn(nn.Module):

    def __init__(self, channel):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Create model
model = catdogcnn(3)

# Load trained weights
model.load_state_dict(
    torch.load(
        "cat_dog_cnn.pth",
        map_location=device
    )
)

# Move model to device
model = model.to(device)

# Evaluation mode
model.eval()