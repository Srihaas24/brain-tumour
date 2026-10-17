import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CLASS_DISPLAY = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'no_tumor': 'No Tumour (Healthy)',
    'pituitary': 'Pituitary Tumour'
}

TUMOR_DETAILS = {
    'glioma': {
        'name': 'Glioma',
        'severity': 'High (Requires Oncological Staging)',
        'origin': 'Glial cells (Astrocytes, Oligodendrocytes, Ependymoma)',
        'description': 'Gliomas originate in the supportive glial cells of the central nervous system. They frequently exhibit diffuse, infiltrative borders with variable surrounding vasogenic edema.',
        'clinical_actions': 'Contrast MRI/Spectroscopy, neurosurgical oncology consult, stereotactic biopsy or maximal safe resection.'
    },
    'meningioma': {
        'name': 'Meningioma',
        'severity': 'Moderate (Typically Benign, Monitor Mass Effect)',
        'origin': 'Meningeal layers (Dura mater / Arachnoid cap cells)',
        'description': 'Meningiomas arise from the meningeal protective coverings of the brain. The majority (~85%) are WHO Grade I slow-growing lesions with distinct dural tails.',
        'clinical_actions': 'High-resolution volumetric MRI follow-up. Microsurgical excision or stereotactic radiosurgery for symptomatic or expanding tumors.'
    },
    'pituitary': {
        'name': 'Pituitary Tumour',
        'severity': 'Moderate (Endocrine & Visual Chiasm Evaluation)',
        'origin': 'Anterior pituitary gland (Sella turcica)',
        'description': 'Pituitary adenomas originate within the sella turcica. They may secrete active hormones (Prolactin, ACTH, GH) or cause bitemporal hemianopsia via optic chiasm compression.',
        'clinical_actions': 'Endocrinological hormonal panel, formal Goldman perimetry visual field mapping, transsphenoidal endoscopic surgery or medical therapy.'
    },
    'no_tumor': {
        'name': 'No Tumour (Normal Brain)',
        'severity': 'Normal / Negative',
        'origin': 'Healthy cerebral parenchyma',
        'description': 'No evidence of intracranial mass lesion, abnormal contrast enhancement, midline shift, or ventricular compression detected in this MRI slice.',
        'clinical_actions': 'Routine screening or symptom correlation if neurological complaints persist.'
    }
}


class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4, backbone='mobilenet_v3'):
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'mobilenet_v3':
            base = models.mobilenet_v3_small(weights=None)
            in_features = base.classifier[0].in_features
            base.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            )
            self.model = base
        else:
            base = models.resnet18(weights=None)
            base.fc = nn.Linear(base.fc.in_features, num_classes)
            self.model = base

    def forward(self, x):
        return self.model(x)


_classifier_cache = {}

def get_classifier(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'model' in _classifier_cache:
        return _classifier_cache['model'], _classifier_cache['is_configured'], device

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    weights_path = os.path.join(base_dir, 'models', 'best_model.pt')

    model = BrainTumorClassifier(num_classes=4, backbone='mobilenet_v3')
    is_configured = False

    if os.path.exists(weights_path):
        try:
            ckpt = torch.load(weights_path, map_location=device)
            state_dict = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state_dict, strict=False)
            is_configured = True
            print(f"[NeuroVision AI] Loaded trained classifier weights from {weights_path}")
        except Exception as e:
            print(f"[NeuroVision AI] Could not load classifier checkpoint: {e}")
            is_configured = False
    else:
        print("[NeuroVision AI] Classifier checkpoint not present. Operating in Demo Output / Unconfigured Mode.")

    model = model.to(device)
    model.eval()
    _classifier_cache['model'] = model
    _classifier_cache['is_configured'] = is_configured
    return model, is_configured, device
