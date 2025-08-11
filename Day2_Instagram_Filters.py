import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

image = torch.randn(3, 100, 100) # 3 RGB Channels and 100 x 100 pixel image
print(f"Image shape: {image.shape}")

#For batch processing we add batch in dimensions too

batch_images = torch.randn(32, 3, 224, 224) # 32 Images
print(f"Batch_shape: {batch_images.shape}")

# Reshaping

original = torch.randn(3, 100, 100)
flattened = original.view(-1) # 1d Tensor for NN
print(f"Flattened: {flattened.shape}")

reshaped = flattened.view(3, 100, 100)
print(f"Back to orginial{reshaped.shape}")

new_shape = original.reshape(30, 10, 100) # Same as view but can handle more edge cases
print(f"New shape: {new_shape.shape}")



"""INDEXING AND SLICING FROM HERE"""



img = torch.randn(3, 200, 200)

red_channel = img[0, :, :]
print(f"Red channel: {red_channel.shape}")

centre_crop = img[:, 50:150, 50:150]  # [All channels, height slice, width slice]
print(f"Cropped{centre_crop.shape}")

downsampled = img[:, ::2, ::2]
print(f"Downsampled: {downsampled.shape}")



"""BROADCASTING"""


img = torch.randn(3, 100, 100)

brightness = 0.3
brighter_img = brightness + img # Will work because broadcasting
print(f"Same shape: {img.shape} == {brighter_img.shape}")

rgb_adjustment = torch.tensor([0.2, -0.1, 0.3]).view(3, 1, 1)
adjusted_img = img + rgb_adjustment
print(f"Per channel adjustment: {adjusted_img.shape}")



""" Real Filter Implementation """

def apply_brightness_filter(image, brightness_factor = 0.3):
    """
    Make the picture brighter
    """
    return torch.clamp(image + brightness_factor, 0 ,1)

def apply_blur_filter(image, kernel_size = 5):
    """
    Blue effect
    """

    kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2) 

    blurred_channels = []
    for i in range(image.shape[0]):
        channel = image[i:i+1].unsqueeze(0) # add batch
        blurred = F.conv2d(channel, kernel, padding=kernel_size // 2)
        blurred_channels.append(blurred.squeeze(0))
    
    return torch.cat(blurred_channels, dim = 0)

def apply_edge_filter(image):
    """
    Edge detection
    """

    #Sobel edge detection filter

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

    # Reshape for conv2d [out_channels, in_channels, kernel_h, kernel_w]
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    edges = []

    for i in range(image.shape[0]):
        channel = image[i:i+1].unsqueeze(0)
        edge_x = F.conv2d(channel, sobel_x, padding=1)
        edge_y = F.conv2d(channel, sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        edges.append(edge_magnitude.squeeze(0))

    return torch.cat(edges, dim = 0)

# Test our filters!
sample_img = torch.rand(3, 200, 200)  # Random image

# Apply filters
bright_img = apply_brightness_filter(sample_img, 0.4)
blur_img = apply_blur_filter(sample_img, kernel_size=7)
edge_img = apply_edge_filter(sample_img)
