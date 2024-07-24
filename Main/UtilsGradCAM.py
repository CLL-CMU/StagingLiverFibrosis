import os
import sys
import argparse
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from os.path import join
from .Model.CoTNet3D_Model import CoTNet3D_Model
from .Datasets.Transforms import read_one_mode_proimg
from .Loss.ordinal_regression import prediction2label
from .UtilsTrain import initialize_model,create_datasets_and_loaders
import torch.nn.functional as F



class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient_maps = None
        self.forward_hook = None
        self.backward_hook = None
        self._register_hooks()  # Register forward and backward hooks
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output  # Capture feature maps from the target layer
        
        def backward_hook(module, grad_in, grad_out):
            self.gradient_maps = grad_out[0]  # Capture gradients from the target layer
        
        # Register hooks to the target layer
        self.forward_hook = self.target_layer.register_forward_hook(forward_hook)
        self.backward_hook = self.target_layer.register_backward_hook(backward_hook)
    
    def _remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
    
    def generate_heatmap(self, input_tensor):
        # Forward pass
        self.model.zero_grad()  # Zero out gradients
        logits = self.model(input_tensor)  # Perform forward pass
        target = prediction2label(logits)  # Get the predicted label index
        print(logits, target)  # Print logits and target for debugging
        
        # Backward pass
        one_hot = torch.zeros_like(logits)  # Create a tensor of zeros with the same shape as logits
        one_hot[0][target] = 1  # Set the entry at the target index to 1
        logits.backward(gradient=one_hot, retain_graph=True)  # Backpropagate using the one-hot tensor
        
        # GradCAM++ calculation
        # Compute weights
        weights = torch.mean(self.gradient_maps, dim=(2, 3, 4), keepdim=True)  # Average gradients across the spatial dimensions
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)  # Weighted sum of feature maps
        cam = F.relu(cam)  # Apply ReLU to the resulting CAM
        
        # Additional computations for GradCAM++
        alpha = weights / (2 * torch.mean(weights, dim=0, keepdim=True) + 1e-5)  # Compute the alpha coefficient
        weights2 = torch.sum(alpha * self.gradient_maps, dim=(2, 3, 4), keepdim=True)  # Second set of weights
        cam_smooth = torch.sum(weights2 * self.feature_maps, dim=1, keepdim=True)  # Generate smooth CAM
        cam_smooth = F.relu(cam_smooth)  # Apply ReLU to the smooth CAM
        
        return cam_smooth  # Return the resulting heatmap


def apply_colormap_on_image(org_img, cam, colormap_name):
    """
    Apply a colormap on the original image with the CAM (Class Activation Map) overlay.
    
    Args:
        org_img (np.ndarray): Original grayscale image.
        cam (np.ndarray): Class Activation Map.
        colormap_name (str): Name of the colormap to apply.
        
    Returns:
        tuple: Overlayed image and colored CAM.
    """
    # Convert CAM to NumPy array and normalize
    cam = cam - np.min(cam)
    cam = cam / np.max(cam) if np.max(cam) > 0 else cam  # Avoid division by zero
    
    # Get the colormap
    colormap = plt.get_cmap(colormap_name)
    cam_colored = colormap(cam)[:, :, :3]  # Take only the RGB channels
    cam_colored = np.uint8(255 * cam_colored)
    
    # Convert grayscale to RGB format
    org_img_rgb = np.repeat(org_img[:, :, np.newaxis], 3, axis=2)
    org_img_rgb = np.uint8((org_img_rgb - np.min(org_img_rgb)) / (np.max(org_img_rgb) - np.min(org_img_rgb)) * 255)
    
    # Overlay CAM on the original image
    overlay = 0.3 * cam_colored + 0.7 * org_img_rgb
    return np.uint8(overlay), np.uint8(cam_colored)


def save_image_with_colorbar(img, cam, colormap_name, save_path):
    """
    Save an image with a colorbar.
    
    Args:
        img (np.ndarray): Original image.
        cam (np.ndarray): Class Activation Map.
        colormap_name (str): Name of the colormap to apply.
        save_path (str): Path to save the image with colorbar.
    """
    fig, ax = plt.subplots()
    cmap = plt.get_cmap(colormap_name)
    
    # Display original image
    ax.imshow(img, cmap='gray', alpha=0.7)
    
    # Display color map
    cax = ax.imshow(cam, cmap=cmap, alpha=0.3)
    
    # Remove axes
    ax.axis('off')
    
    # Add color bar
    cbar = fig.colorbar(cax, ax=ax, ticks=[np.min(cam), np.max(cam)])
    cbar.ax.set_yticklabels(['Low', 'High'])  # Set labels for colorbar
    cbar.solids.set_edgecolor('face')  # Ensure colorbar colors are saturated
    
    # Save image
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_gradcam_gif(org_img_tensor, grad_cam_tensor, filepath, colormap_name='jet'):
    """
    Save a series of GradCAM images as a GIF.
    
    Args:
        org_img_tensor (torch.Tensor): Original 3D medical image tensor.
        grad_cam_tensor (torch.Tensor): GradCAM 3D tensor.
        filepath (str): Path to save the GIF.
        colormap_name (str): Name of the colormap to apply.
    """
    # Ensure tensor is on CPU
    org_img_tensor = org_img_tensor.cpu()
    grad_cam_tensor = grad_cam_tensor.cpu()
    
    # Prepare frames for the GIF
    frames1 = []
    frames2 = []
    for t in range(org_img_tensor.size(1)):
        # Get single channel image and convert to NumPy array
        org_img_slice = org_img_tensor[0, t].squeeze().detach().numpy()  # Shape: (H, W)
        grad_cam_slice = grad_cam_tensor[0, t].squeeze().detach().numpy()  # Shape: (H, W)

        # Overlay CAM on the original image
        overlayed_img, cam_colored = apply_colormap_on_image(org_img_slice, grad_cam_slice, colormap_name)
        frames1.append(overlayed_img)
        frames2.append(cam_colored)
        
        save_image_with_colorbar(org_img_slice, grad_cam_slice, colormap_name, save_path=join(filepath, f'frame_{t}.png'))
    
    # Save frames as GIF
    imageio.mimsave(join(filepath, 'overlayed_img.gif'), frames1, duration=0.5)  # 0.5 seconds per frame
    imageio.mimsave(join(filepath, 'cam_colored.gif'), frames2, duration=0.5)  # 0.5 seconds per frame


def grad_cam_3d_gif(model, org_img_tensor, gitpath):
    """
    Generate and save a GradCAM++ GIF for a 3D image.
    
    Args:
        model (torch.nn.Module): The model to use for GradCAM++.
        org_img_tensor (torch.Tensor): Original 3D image tensor.
        gitpath (str): Path to save the GIF.
        
    Returns:
        torch.Tensor: The original image tensor, possibly modified.
    """

    gradcam = GradCAMpp(model, target_layer=model.layer2)
    
    # Generate GradCAM++ heatmap
    heatmap_tensor = gradcam.generate_heatmap(org_img_tensor)
    
    # Get output size
    output_size = org_img_tensor.size()[2:]
    
    # Upsample heatmap to match the original image size
    upsampled_heatmap = F.interpolate(heatmap_tensor, size=output_size, mode='trilinear', align_corners=False).squeeze()
    
    change_img_tensor = org_img_tensor.squeeze(0)
    upsampled_heatmap = upsampled_heatmap.unsqueeze(0)
    save_gradcam_gif(change_img_tensor, upsampled_heatmap, gitpath, colormap_name='jet')
    
    return change_img_tensor


def Infer_GradCAM_Main(args):
    """
    Main function for performing inference with GradCAM++.
    
    Args:
        args (argparse.Namespace): Configuration parameters.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model and load it to the specified device
    model, _, _, _ = initialize_model(args, device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Read the image data and convert it into a tensor
    img_data = read_one_mode_proimg(args.imagepath, args.target_shapes)
    input_tensor = torch.from_numpy(img_data).unsqueeze(1)

    # Move the tensor to the specified device
    input_tensor = input_tensor.to(device)
    # print(input_tensor.shape)
    
    # Prepare the output path for saving the results
    output_path = f'{args.camoutput}/{args.mode}/'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate GradCAM++ and save as GIF
    grad_cam_3d_gif(model, input_tensor, gitpath=output_path)





