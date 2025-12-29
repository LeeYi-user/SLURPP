import torch
from PIL import Image


def save_image(path, tensor, original_size=None):
    tensor = tensor.squeeze(0).cpu()
    tensor = torch.clamp(tensor, 0, 1)

    image = None

    if tensor.size(0) == 3:
        # Convert 3-channel tensor (C, H, W) to (H, W, C)
        tensor = tensor.permute(1, 2, 0).numpy()
        tensor = (tensor * 255).astype('uint8')
        image = Image.fromarray(tensor)  # Create an RGB image
        
        # Resize to original size if provided
        if original_size is not None:
            image = image.resize(original_size, Image.LANCZOS)
        
        image.save(path)

    elif tensor.size(0) == 1:
        # Convert 1-channel tensor to grayscale (H, W)
        tensor = tensor.squeeze(0).numpy()
        tensor = (tensor * 255).astype('uint8')
        image = Image.fromarray(tensor, mode='L')  # Create a grayscale image
        
        # Resize to original size if provided
        if original_size is not None:
            image = image.resize(original_size, Image.LANCZOS)
        
        image.save(path)
    else:
        pass


def normalize_imgs(rgb, gamma = None, device = "cuda", dtype = torch.float32):
    rgb = torch.clamp(rgb, 0, 1)
    if gamma is not None:
        rgb = rgb ** gamma
    rgb_norm: torch.Tensor = rgb * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
    rgb_norm = rgb_norm.to(device).to(dtype)
    assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

    return rgb_norm
