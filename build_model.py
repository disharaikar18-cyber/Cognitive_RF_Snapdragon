import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class RFClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(RFClassifier, self).__init__()
        self.backbone = mobilenet_v3_small(pretrained=False)
        self.backbone.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.backbone.classifier[3] = nn.Linear(self.backbone.classifier[3].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def export_onnx():
    model = RFClassifier(num_classes=3)
    model.eval()
    dummy_input = torch.randn(1, 1, 64, 64)
    
    # Export ensuring all weights are embedded inside a single ONNX file
    torch.onnx.export(
        model,
        dummy_input,
        "rf_classifier.onnx",
        input_names=["spectrogram"],
        output_names=["logits"],
        opset_version=17,
        export_params=True
    )
    print("Model successfully exported to self-contained rf_classifier.onnx")

if __name__ == "__main__":
    export_onnx()