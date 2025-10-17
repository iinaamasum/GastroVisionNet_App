from typing import Tuple

import numpy as np
import torch
import cv2


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        # full backward hook is required for PyTorch 2.x
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= weights[i]
        heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        denom = np.max(heatmap) + 1e-10
        heatmap /= denom
        return heatmap


def overlay_heatmap(
    original_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    heatmap_resized = cv2.resize(
        heatmap, (original_bgr.shape[1], original_bgr.shape[0])
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = color_map * alpha + original_bgr
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay
