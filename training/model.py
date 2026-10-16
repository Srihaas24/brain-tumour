"""
Deep Learning Model Architecture for Brain Tumor Classification & Explainability (Grad-CAM).
Supports MobileNetV3-Small (fast, lightweight, highly accurate) and custom CNN architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CLASS_DISPLAY = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'no_tumor': 'No Tumor (Healthy)',
    'pituitary': 'Pituitary Tumor'
}

TUMOR_INFO = {
    'glioma': {
        'name': 'Glioma',
        'has_tumor': True,
        'severity': 'High (Requires Oncological Staging)',
        'origin': 'Glial cells (astrocytes, oligodendrocytes, ependymal cells)',
        'description': 'Gliomas originate in the supportive glial cells of the brain. They can range from slow-growing low-grade astrocytomas to aggressive high-grade glioblastomas (GBM). They frequently exhibit diffuse margins and infiltrative borders.',
        'clinical_actions': 'Contrast MRI/Spectroscopy, neurosurgical consult, biopsy, potential surgical resection followed by radiation therapy and Temozolomide chemotherapy.'
    },
    'meningioma': {
        'name': 'Meningioma',
        'has_tumor': True,
        'severity': 'Moderate (Typically Benign, Watch for Mass Effect)',
        'origin': 'Meninges (dura mater, arachnoid mater)',
        'description': 'Meningiomas arise from the meningeal protective layers surrounding the brain and spinal cord. Approximately 85-90% are benign (WHO Grade I), well-circumscribed, and slow-growing, though they can cause symptoms through compression of adjacent brain tissue.',
        'clinical_actions': 'High-resolution volumetric MRI follow-up. For symptomatic or expanding lesions, complete microsurgical resection (Simpson grading) or stereotactic radiosurgery (Gamma Knife).'
    },
    'pituitary': {
        'name': 'Pituitary Tumor',
        'has_tumor': True,
        'severity': 'Moderate (Endocrine & Visual Field Evaluation Needed)',
        'origin': 'Pituitary gland (sella turcica at skull base)',
        'description': 'Pituitary adenomas arise in the pituitary gland nestled in the sella turcica. Most are benign but can secrete excess hormones (prolactin, ACTH, GH) or cause bitemporal hemianopsia through mass-effect compression of the optic chiasm.',
        'clinical_actions': 'Complete endocrinology panel (prolactin, IGF-1, cortisol), formal visual field perimetry testing, transsphenoidal endoscopic surgical removal or dopamine agonist therapy (e.g., Cabergoline).'
    },
    'no_tumor': {
        'name': 'No Tumor (Normal Scan)',
        'has_tumor': False,
        'severity': 'Normal / Negative',
        'origin': 'Healthy cerebral parenchyma',
        'description': 'No evidence of intracranial mass, abnormal contrast enhancement, abnormal midline shift, or ventricular distortion detected in this MRI slice. Parenchyma, sulci, and gyri appear normal for age.',
        'clinical_actions': 'Routine follow-up or symptom correlation if clinical neurological deficits persist.'
    }
}


class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, backbone='mobilenet_v3'):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == 'mobilenet_v3':
            base = None
            if pretrained:
                try:
                    weights = models.MobileNet_V3_Small_Weights.DEFAULT
                    base = models.mobilenet_v3_small(weights=weights)
                except Exception as e:
                    print(f"Note: Pretrained weights download failed ({e}). Initializing with weights=None.")
                    base = models.mobilenet_v3_small(weights=None)
            else:
                base = models.mobilenet_v3_small(weights=None)
                
            self.features = base.features
            in_features = base.classifier[0].in_features
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes)
            )
            # The last conv layer for Grad-CAM
            self.target_layer = self.features[-1]
            
        elif backbone == 'resnet18':
            base = None
            if pretrained:
                try:
                    weights = models.ResNet18_Weights.DEFAULT
                    base = models.resnet18(weights=weights)
                except Exception as e:
                    print(f"Note: Pretrained weights download failed ({e}). Initializing with weights=None.")
                    base = models.resnet18(weights=None)
            else:
                base = models.resnet18(weights=None)

            self.features = nn.Sequential(*list(base.children())[:-2])
            in_features = base.fc.in_features
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes)
            )
            self.target_layer = self.features[-1]

        else:
            # Custom lightweight CNN fallback
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((7, 7))
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 7 * 7, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes)
            )
            self.target_layer = self.features[-4]

        # Hooks storage for Grad-CAM
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out

    def generate_gradcam(self, input_tensor, target_class=None):
        """
        Generates Grad-CAM activation heatmap for the input tensor.
        input_tensor shape: (1, 3, 224, 224)
        """
        self.eval()
        self.zero_grad()
        
        logits = self.forward(input_tensor)
        probs = F.softmax(logits, dim=1)
        
        if target_class is None:
            target_class = torch.argmax(probs, dim=1).item()
            
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        # Pool the gradients across the channels
        if self.gradients is None or self.activations is None:
            return None, probs[0].detach().cpu().numpy()
            
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0].detach()
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU
        max_val = np.max(heatmap)
        if max_val > 1e-8:
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)
            
        return heatmap, probs[0].detach().cpu().numpy(), target_class
