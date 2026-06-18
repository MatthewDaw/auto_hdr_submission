"""Export the trained color embedding to ONNX for browser (onnxruntime-web)."""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

class Model(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        m = mobilenet_v3_small(weights=None)
        self.backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(576,256), nn.ReLU(), nn.Linear(256,dim))
    def forward(self,x): return F.normalize(self.proj(self.backbone(x)),dim=1)

m = Model(); m.load_state_dict(torch.load("embed2_best.pt", map_location="cpu")); m.eval()
dummy = torch.randn(1,3,128,128)
torch.onnx.export(m, dummy, "embed_color.onnx",
                  input_names=["lab_norm"], output_names=["embedding"],
                  dynamic_axes={"lab_norm":{0:"batch"}, "embedding":{0:"batch"}},
                  opset_version=17, dynamo=False)  # legacy exporter: single self-contained file
import os
print(f"exported embed_color.onnx ({os.path.getsize('embed_color.onnx')/1e6:.1f} MB)")
print("input: (B,3,128,128) Lab/255, L=CLAHE'd, normalized (x-0.5)/0.25 ; output: (B,128) L2-normalized")
