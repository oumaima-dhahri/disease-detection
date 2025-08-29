import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


MODEL_PATH = '../saved_models_and_data/wheat_disease_hybrid_model.pth'
DATASET_DIR = '../dataset'
TEST_IMAGES_DIR = '../test_images'
IMAGE_SIZE = (224, 224)


class HybridModel(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.efficientnet_features = nn.Sequential(*list(self.efficientnet.children())[:-1])
        resnet_features = 2048
        efficientnet_features = 1280
        fusion_dim = 512
        self.fusion = nn.Sequential(
            nn.Linear(resnet_features + efficientnet_features, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)
    def forward(self, x):
        resnet_features = self.resnet_features(x).flatten(1)
        efficientnet_features = self.efficientnet_features(x).flatten(1)
        combined_features = torch.cat([resnet_features, efficientnet_features], dim=1)
        fused_features = self.fusion(combined_features)
        output = self.classifier(fused_features)
        return output


def get_class_labels(dataset_dir):
    return sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
class_labels = get_class_labels(DATASET_DIR)
test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridModel(num_classes=len(class_labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))
    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))
        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, IMAGE_SIZE)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
def compute_saliency_map(model, input_tensor, class_idx=None):
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    model.zero_grad()
    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    loss = output[0, class_idx]
    loss.backward()
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)
    saliency = cv2.resize(saliency, IMAGE_SIZE)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency
def compute_attention_map(model, input_tensor):
    with torch.no_grad():
        features = None
        def hook_fn(module, input, output):
            nonlocal features
            features = output.detach()
        handle = model.resnet.layer4[-1].register_forward_hook(hook_fn)
        _ = model(input_tensor)
        handle.remove()
        attn_map = features.mean(dim=1).squeeze().cpu().numpy()
        attn_map = cv2.resize(attn_map, IMAGE_SIZE)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map
def show_cam_on_image(img: np.ndarray, mask: np.ndarray, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
target_layer = model.resnet.layer4[-1]
gradcam = GradCAM(model, target_layer)
for img_name in os.listdir(TEST_IMAGES_DIR):
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = test_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)[0]
        pred_idx = prob.argmax().item()
        pred_label = class_labels[pred_idx]
        pred_prob = prob[pred_idx].item()
    cam = gradcam(img_tensor, class_idx=pred_idx)
    saliency = compute_saliency_map(model, img_tensor, class_idx=pred_idx)
    attn_map = compute_attention_map(model, img_tensor)
    img_np = np.array(img_pil.resize(IMAGE_SIZE)).astype(np.float32) / 255.0
    cam_img = show_cam_on_image(img_np, cam)
    sal_img = show_cam_on_image(img_np, saliency)
    attn_img = show_cam_on_image(img_np, attn_map)
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 5, 1)
    plt.imshow(img_pil.resize(IMAGE_SIZE))
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 5, 2)
    plt.imshow(cam, cmap='jet')
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.subplot(1, 5, 3)
    plt.imshow(cam_img)
    plt.title('Grad-CAM Overlay')
    plt.axis('off')
    plt.subplot(1, 5, 4)
    plt.imshow(sal_img)
    plt.title('Saliency Overlay')
    plt.axis('off')
    plt.subplot(1, 5, 5)
    plt.imshow(attn_img)
    plt.title('Attention Map')
    plt.axis('off')
    plt.suptitle(f'Image: {img_name}\nPred: {pred_label} (Prob: {pred_prob:.2f})')
    plt.tight_layout()
    plt.show()
