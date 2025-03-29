import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load pre-trained ResNet model
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
image_path = 'GradCAM-Dataset/cat_dog.jpg'
image = Image.open(image_path)
image_tensor = preprocess(image)
input_batch = image_tensor.unsqueeze(0)

# Get the model's prediction
output = model(input_batch)
predicted_class = output.argmax().item()

# Get the feature maps from the last convolutional layer
feature_maps = None
def hook_feature(module, input, output):
    global feature_maps
    feature_maps = output

model.layer4.register_forward_hook(hook_feature)

# Get the gradient of the output with respect to the feature maps
output[:, predicted_class].backward()
gradients = feature_maps.grad

# Calculate the importance weights
weights = torch.mean(gradients, dim=(2, 3))

# Generate the class activation map
class_activation_map = torch.zeros(feature_maps.shape[2:], dtype=torch.float32)
for i, w in enumerate(weights[0]):
    class_activation_map += w * feature_maps[0, i].detach()

# Normalize the class activation map
class_activation_map = F.relu(class_activation_map)
class_activation_map = class_activation_map.numpy()
class_activation_map = cv2.resize(class_activation_map, (224, 224))
class_activation_map = (class_activation_map - class_activation_map.min()) / (class_activation_map.max() - class_activation_map.min())

# Convert the original image to numpy array
original_image = np.array(image.resize((224, 224)))

# Create a heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * class_activation_map), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Combine the original image with the heatmap
result = heatmap * 0.3 + original_image * 0.7

# Display the results
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(heatmap)
plt.title('Heatmap')
plt.axis('off')

plt.subplot(133)
plt.imshow(result.astype(np.uint8))
plt.title('GradCAM Result')
plt.axis('off')

plt.tight_layout()
plt.show()
