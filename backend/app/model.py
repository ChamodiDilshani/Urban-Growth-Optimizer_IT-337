"""
Model definition + loader.

The architecture is copied from the notebook (cell "SiameseResUNet") so the
raw state-dict file `best_siam_resunet.pth` can be used as a fallback.
The preferred file is the TorchScript export `siam_resunet_complete.pt`,
which needs no class definition at all.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Architecture (identical to the notebook) ----------------
def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class FuseBlock(nn.Module):
    """Bi-temporal fusion: [f1, f2, |f1-f2|] -> conv"""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(conv_bn_relu(3 * c, c), conv_bn_relu(c, c))

    def forward(self, f1, f2):
        return self.conv(torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(conv_bn_relu(cin + cskip, cout), conv_bn_relu(cout, cout))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SiameseResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights=None)   # weights come from the checkpoint
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        chans = [64, 64, 128, 256, 512]
        self.fuse = nn.ModuleList([FuseBlock(c) for c in chans])
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.dec0 = DecoderBlock(64, 64, 32)
        self.head = nn.Sequential(conv_bn_relu(32, 16), nn.Conv2d(16, 1, 1))

    def encode(self, x):
        f0 = self.stem(x)
        f1 = self.layer1(self.pool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]

    def forward(self, a, b):
        fa, fb = self.encode(a), self.encode(b)
        z = [fuse(x, y) for fuse, x, y in zip(self.fuse, fa, fb)]
        d = self.dec3(z[4], z[3])
        d = self.dec2(d, z[2])
        d = self.dec1(d, z[1])
        d = self.dec0(d, z[0])
        out = self.head(d)
        return F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)


# ---------------- Loading ----------------
def load_model():
    """Returns (model, source_filename). Prefers the TorchScript export."""
    pt_path = MODELS_DIR / "siam_resunet_complete.pt"
    pth_path = MODELS_DIR / "best_siam_resunet.pth"

    if pt_path.exists():
        model = torch.jit.load(str(pt_path), map_location=DEVICE)
        model.eval()
        return model, pt_path.name

    if pth_path.exists():
        model = SiameseResUNet().to(DEVICE)
        model.load_state_dict(torch.load(str(pth_path), map_location=DEVICE))
        model.eval()
        return model, pth_path.name

    raise FileNotFoundError(
        f"No model file found in {MODELS_DIR}. Copy siam_resunet_complete.pt "
        f"(or best_siam_resunet.pth) from the zip into that folder."
    )


def load_threshold(default=0.5):
    p = MODELS_DIR / "threshold.json"
    if p.exists():
        return float(json.loads(p.read_text())["threshold"])
    return default


def load_test_metrics():
    p = MODELS_DIR / "test_metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return None
