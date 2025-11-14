"""Quick test to inspect extracted weight shapes"""
import torch

weights_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/pyfaceau/detectors/openface_mtcnn_weights.pth"
state_dict = torch.load(weights_path, map_location='cpu')

print("PNet weights:")
for key, value in state_dict['pnet'].items():
    print(f"  {key:20s} {tuple(value.shape)}")

print("\nRNet weights:")
for key, value in state_dict['rnet'].items():
    print(f"  {key:20s} {tuple(value.shape)}")

print("\nONet weights:")
for key, value in state_dict['onet'].items():
    print(f"  {key:20s} {tuple(value.shape)}")
