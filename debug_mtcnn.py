"""Debug MTCNN step by step"""
import torch
import numpy as np
import cv2
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN, PNet, RNet, ONet

print("Step 1: Load weights")
weights_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/pyfaceau/detectors/openface_mtcnn_weights.pth"
state_dict = torch.load(weights_path, map_location='cpu')
print("  ✓ Weights loaded")

print("\nStep 2: Create PNet")
pnet = PNet()
print("  ✓ PNet created")

print("\nStep 3: Load PNet weights")
pnet_state = state_dict['pnet']
pnet.conv1.weight.data = pnet_state['conv1.weight']
pnet.conv1.bias.data = pnet_state['conv1.bias'].float()
pnet.prelu1.weight.data = pnet_state['prelu1.weight']

pnet.conv2.weight.data = pnet_state['conv2.weight']
pnet.conv2.bias.data = pnet_state['conv2.bias'].float()
pnet.prelu2.weight.data = pnet_state['prelu2.weight']

pnet.conv3.weight.data = pnet_state['conv3.weight']
pnet.conv3.bias.data = pnet_state['conv3.bias'].float()
pnet.prelu3.weight.data = pnet_state['prelu3.weight']

fc_weight = pnet_state['fc1.weight']
fc_bias = pnet_state['fc1.bias']
pnet.conv4_1.weight.data = fc_weight[:2, :].unsqueeze(-1).unsqueeze(-1)
pnet.conv4_1.bias.data = fc_bias[:2]
pnet.conv4_2.weight.data = fc_weight[2:6, :].unsqueeze(-1).unsqueeze(-1)
pnet.conv4_2.bias.data = fc_bias[2:6]
print("  ✓ PNet weights loaded")

print("\nStep 4: Test PNet forward pass")
pnet.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 64, 64)
    cls, bbox = pnet(x)
    print(f"  Input: {x.shape}")
    print(f"  Cls: {cls.shape}")
    print(f"  Bbox: {bbox.shape}")
print("  ✓ PNet forward pass works")

print("\nStep 5: Initialize full detector")
try:
    detector = OpenFaceMTCNN()
    print("  ✓ Detector initialized")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nStep 6: Create test image")
test_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
print(f"  ✓ Test image: {test_img.shape}")

print("\nStep 7: Preprocess image")
img_preprocessed = detector._preprocess_image(test_img)
print(f"  ✓ Preprocessed: {img_preprocessed.shape}, dtype={img_preprocessed.dtype}")
print(f"  Value range: [{img_preprocessed.min():.3f}, {img_preprocessed.max():.3f}]")

print("\nStep 8: Calculate pyramid scales")
scales = detector._calculate_scales(480, 640)
print(f"  ✓ Scales: {len(scales)} levels")
print(f"  Scale values: {scales[:3]}")

print("\nStep 9: Test detection on single scale")
try:
    scale = scales[0]
    hs = int(480 * scale)
    ws = int(640 * scale)
    print(f"  Testing scale {scale:.3f} -> {ws}x{hs}")

    img_scaled = cv2.resize(img_preprocessed, (ws, hs))
    print(f"  Scaled image: {img_scaled.shape}")

    img_tensor = torch.from_numpy(img_scaled).permute(2, 0, 1).unsqueeze(0).float()
    print(f"  Tensor: {img_tensor.shape}")

    with torch.no_grad():
        cls, reg = detector.pnet(img_tensor)
        print(f"  PNet output: cls={cls.shape}, reg={reg.shape}")

    cls_np = torch.nn.functional.softmax(cls, dim=1).cpu().numpy()[0, 1, :, :]
    reg_np = reg.cpu().numpy()[0, :, :, :]
    print(f"  Numpy: cls={cls_np.shape}, reg={reg_np.shape}")
    print(f"  Cls range: [{cls_np.min():.3f}, {cls_np.max():.3f}]")

    print("  ✓ Single scale detection works")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nStep 10: Test full detection pipeline")
try:
    bboxes, landmarks = detector.detect(test_img, return_landmarks=True)
    print(f"  ✓ Detection completed")
    print(f"  Detected {len(bboxes)} faces")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("All steps completed successfully!")
print("="*60)
