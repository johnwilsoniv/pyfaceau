"""Simple test to initialize models with OpenFace weights"""
import torch
import torch.nn as nn

# Simple PNet test
class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.prelu3 = nn.PReLU(32)

        # FC layer - needs special handling for fully conv
        # Original: (6, 32) means output_dim=6, input_dim=32
        # For fully conv: need to reshape to 1x1 conv
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)  # classification
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)  # bbox

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.prelu2(x)

        x = self.conv3(x)
        x = self.prelu3(x)

        cls = self.conv4_1(x)
        bbox = self.conv4_2(x)

        return cls, bbox

print("Loading weights...")
weights_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/pyfaceau/detectors/openface_mtcnn_weights.pth"
state_dict = torch.load(weights_path, map_location='cpu')

print("\nCreating PNet...")
pnet = PNet()

print("\nLoading PNet weights manually...")
# Load conv layers
pnet.conv1.weight.data = state_dict['pnet']['conv1.weight']
pnet.conv1.bias.data = state_dict['pnet']['conv1.bias'].float()
pnet.prelu1.weight.data = state_dict['pnet']['prelu1.weight']

pnet.conv2.weight.data = state_dict['pnet']['conv2.weight']
pnet.conv2.bias.data = state_dict['pnet']['conv2.bias'].float()
pnet.prelu2.weight.data = state_dict['pnet']['prelu2.weight']

pnet.conv3.weight.data = state_dict['pnet']['conv3.weight']
pnet.conv3.bias.data = state_dict['pnet']['conv3.bias'].float()
pnet.prelu3.weight.data = state_dict['pnet']['prelu3.weight']

# Load FC layer (6, 32) -> split into cls (2, 32) and bbox (4, 32)
# Reshape to (out, in, 1, 1) for 1x1 conv
fc_weight = state_dict['pnet']['fc1.weight']  # (6, 32)
fc_bias = state_dict['pnet']['fc1.bias']  # (6,)

pnet.conv4_1.weight.data = fc_weight[:2, :].unsqueeze(-1).unsqueeze(-1)  # (2, 32, 1, 1)
pnet.conv4_1.bias.data = fc_bias[:2]

pnet.conv4_2.weight.data = fc_weight[2:6, :].unsqueeze(-1).unsqueeze(-1)  # (4, 32, 1, 1)
pnet.conv4_2.bias.data = fc_bias[2:6]

print("PNet loaded successfully!")

# Test forward pass
print("\nTesting forward pass...")
x = torch.randn(1, 3, 64, 64)
cls, bbox = pnet(x)
print(f"Input: {x.shape}")
print(f"Cls output: {cls.shape}")
print(f"Bbox output: {bbox.shape}")

print("\nSuccess!")
