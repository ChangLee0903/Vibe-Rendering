# ============================================================
# Vibe Optimization (CLIP + VLM SceneParser + VLM Confiner) — GPT-4o
# NaN-safe, NOT conservative:
# - optimize ALL params every outer
# - BG params updated every outer
# - adds: CLIP float32, grad clipping, nan_to_num guards
# ============================================================

import os, io, json, base64, re
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

import clip
from openai import OpenAI

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    Materials,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
JsonDict = Dict[str, Any]

IMAGE_SIZE = int(os.getenv("VIBE_IMAGE_SIZE", "512"))
STEPS_OUTER = int(os.getenv("VIBE_STEPS_OUTER", "6"))
STEPS_INNER = int(os.getenv("VIBE_STEPS_INNER", "40"))
LR = float(os.getenv("VIBE_LR", "0.02"))

# not conservative weights
W_FG = float(os.getenv("W_FG", "1.0"))
W_BG = float(os.getenv("W_BG", "1.0"))
W_FULL = float(os.getenv("W_FULL", "0.25"))
W_NEG = float(os.getenv("W_NEG", "0.8"))
W_BG_COLOR = float(os.getenv("W_BG_COLOR", "0.6"))

NEG_MARGIN = float(os.getenv("NEG_MARGIN", "0.25"))
GRAD_CLIP = float(os.getenv("GRAD_CLIP", "1.0"))  # <- key for NaN stability
MAX_OPT_NUM = int(os.getenv("MAX_OPT_NUM", "8"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("[warn] OPENAI_API_KEY is empty. VLM parts will fail unless set.")

BG_PARAMS = {
    "fog_color", "fog_density", "fog_falloff", "fog_height_bias",
    "exposure", "gamma", "contrast", "hue_shift", "saturation", "vignette_strength",
    "ambient_color", "light_color"
}

# -----------------------------
# Utils
# -----------------------------

def save_render(rgb_t: torch.Tensor, path: str):
    img = (torch.nan_to_num(rgb_t, nan=0.0, posinf=1.0, neginf=0.0)
           .detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    pil = Image.fromarray(img)

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pil.save(path)

def safe_nan_to_num_(t: torch.Tensor, nan: float = 0.0, posinf: float = 1.0, neginf: float = 0.0):
    # inplace nan_to_num
    if torch.is_floating_point(t):
        torch.nan_to_num_(t, nan=nan, posinf=posinf, neginf=neginf)

def show_progress(img_t: torch.Tensor, params: dict, title: str = ""):
    # NOTE: keep signature unchanged; we just save rendered result only.
    img = (torch.nan_to_num(img_t, nan=0.0, posinf=1.0, neginf=0.0)
           .detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    safe = title.replace("/", "_").replace(" ", "_") if title else "render"
    plt.savefig(f"{safe}.png", dpi=140, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

# -----------------------------
# Color hints
# -----------------------------
COLOR_HINTS = {
    "navy blue": (0.05, 0.07, 0.20),
    "navy": (0.05, 0.07, 0.20),
    "dark blue": (0.05, 0.10, 0.35),
    "blue": (0.10, 0.25, 0.85),
    "black": (0.02, 0.02, 0.02),
    "white": (0.98, 0.98, 0.98),
    "pink": (1.00, 0.30, 0.70),
    "magenta": (0.95, 0.10, 0.85),
    "purple": (0.55, 0.20, 0.80),
    "violet": (0.55, 0.20, 0.80),
}

def _find_color_phrase(text: str):
    t = text.lower()
    for k in sorted(COLOR_HINTS.keys(), key=lambda s: -len(s)):
        if k in t:
            return k, COLOR_HINTS[k]
    return None

def get_bg_color_target(prompt: str) -> Optional[torch.Tensor]:
    t = prompt.lower()
    m = re.search(r"(.{0,80})background", t)
    if m:
        hit = _find_color_phrase(m.group(0))
        if hit:
            return torch.tensor(hit[1], device=device, dtype=torch.float32)
    hit = _find_color_phrase(prompt)
    if hit:
        return torch.tensor(hit[1], device=device, dtype=torch.float32)
    return None

# -----------------------------
# Params
# -----------------------------
class AdvancedVibeParams(nn.Module):
    def __init__(self, device=device):
        super().__init__()

        self.fog_density = nn.Parameter(torch.tensor(0.05, device=device))
        self.fog_color = nn.Parameter(torch.tensor([0.30, 0.30, 0.30], device=device))
        self.fog_falloff = nn.Parameter(torch.tensor(1.0, device=device))
        self.fog_height_bias = nn.Parameter(torch.tensor(0.0, device=device))

        self.light_intensity = nn.Parameter(torch.tensor(1.2, device=device))
        self.light_color = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], device=device))
        self.light_dir = nn.Parameter(torch.tensor([0.3, -0.8, -0.4], device=device))
        self.rim_intensity = nn.Parameter(torch.tensor(0.2, device=device))
        self.rim_color = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], device=device))

        self.ambient_color = nn.Parameter(torch.tensor([0.10, 0.10, 0.12], device=device))

        self.camera_distance = nn.Parameter(torch.tensor(2.5, device=device))
        self.camera_elev = nn.Parameter(torch.tensor(20.0, device=device))
        self.camera_azim = nn.Parameter(torch.tensor(45.0, device=device))
        self.fov = nn.Parameter(torch.tensor(60.0, device=device))

        self.diffuse_tint = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], device=device))
        self.specular_strength = nn.Parameter(torch.tensor(0.3, device=device))
        self.specular_color = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], device=device))
        self.roughness = nn.Parameter(torch.tensor(0.5, device=device))
        self.shininess = nn.Parameter(torch.tensor(32.0, device=device))

        self.contrast = nn.Parameter(torch.tensor(1.0, device=device))
        self.exposure = nn.Parameter(torch.tensor(0.0, device=device))
        self.gamma = nn.Parameter(torch.tensor(1.0, device=device))
        self.saturation = nn.Parameter(torch.tensor(1.0, device=device))
        self.hue_shift = nn.Parameter(torch.tensor(0.0, device=device))
        self.vignette_strength = nn.Parameter(torch.tensor(0.0, device=device))

        self.dynamic_ranges: Dict[str, Any] = {}

    def update_range(self, name: str, rng: Any):
        self.dynamic_ranges[name] = rng

    def get_state_values(self) -> Dict[str, Any]:
        return {k: v.detach().cpu().tolist() for k, v in self.named_parameters()}

    def clamp_(self):
        with torch.no_grad():
            # nan guard first
            for _, p in self.named_parameters():
                safe_nan_to_num_(p, nan=0.0, posinf=1.0, neginf=0.0)

            for name, p in self.named_parameters():
                # apply dynamic range first if exists
                if name in self.dynamic_ranges:
                    r = self.dynamic_ranges[name]
                    if p.ndim == 1 and p.shape[0] == 3 and isinstance(r, list) and len(r) == 2 and isinstance(r[0], list):
                        for i in range(3):
                            p.data[i].clamp_(float(r[0][i]), float(r[1][i]))
                    elif isinstance(r, list) and len(r) == 2 and all(isinstance(x, (int, float)) for x in r):
                        p.data.clamp_(float(r[0]), float(r[1]))
                    continue

                # defaults
                if name in ["fog_density", "specular_strength", "vignette_strength", "roughness"]:
                    p.data.clamp_(0.0, 1.0)
                elif name == "shininess":
                    p.data.clamp_(1.0, 128.0)
                elif name == "contrast":
                    p.data.clamp_(0.5, 2.0)
                elif name == "exposure":
                    p.data.clamp_(-2.0, 2.0)
                elif name == "gamma":
                    p.data.clamp_(0.5, 2.5)
                elif name == "saturation":
                    p.data.clamp_(0.0, 2.0)
                elif name == "hue_shift":
                    p.data.clamp_(-0.25, 0.25)
                elif name == "fog_falloff":
                    p.data.clamp_(0.1, 5.0)
                elif name == "fog_height_bias":
                    p.data.clamp_(-1.0, 1.0)
                elif name == "light_intensity":
                    p.data.clamp_(0.0, 5.0)
                elif name == "rim_intensity":
                    p.data.clamp_(0.0, 3.0)
                elif name == "camera_distance":
                    p.data.clamp_(1.0, 6.0)
                elif name == "camera_elev":
                    p.data.clamp_(-80.0, 80.0)
                elif name == "camera_azim":
                    p.data.clamp_(-180.0, 180.0)
                elif name == "fov":
                    p.data.clamp_(10.0, 120.0)
                elif "color" in name:
                    p.data.clamp_(0.0, 1.5)
                elif "tint" in name:
                    p.data.clamp_(0.0, 2.0)
                elif name == "light_dir":
                    p.data.clamp_(-1.0, 1.0)

            # nan guard again
            for _, p in self.named_parameters():
                safe_nan_to_num_(p, nan=0.0, posinf=1.0, neginf=0.0)

# -----------------------------
# HSV (no inplace)
# -----------------------------
def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    rgb = torch.clamp(rgb, 0.0, 1.0)
    r, g, b = rgb.unbind(-1)
    maxc = torch.max(rgb, dim=-1).values
    minc = torch.min(rgb, dim=-1).values
    v = maxc
    delt = (maxc - minc).clamp_min(1e-8)
    s = delt / (maxc + 1e-8)

    rc = (maxc - r) / delt
    gc = (maxc - g) / delt
    bc = (maxc - b) / delt

    h = torch.zeros_like(maxc)
    h = torch.where((maxc == r), (bc - gc), h)
    h = torch.where((maxc == g), (2.0 + rc - bc), h)
    h = torch.where((maxc == b), (4.0 + gc - rc), h)
    h = (h / 6.0) % 1.0
    out = torch.stack([h, s, v], dim=-1)
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return out

def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    hsv = torch.nan_to_num(hsv, nan=0.0, posinf=1.0, neginf=0.0)
    h, s, v = hsv.unbind(-1)
    h6 = h * 6.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    c0 = torch.stack([v, t, p], dim=-1)
    c1 = torch.stack([q, v, p], dim=-1)
    c2 = torch.stack([p, v, t], dim=-1)
    c3 = torch.stack([p, q, v], dim=-1)
    c4 = torch.stack([t, p, v], dim=-1)
    c5 = torch.stack([v, p, q], dim=-1)

    oh = F.one_hot(i, num_classes=6).to(hsv.dtype)
    out = (
        c0 * oh[..., 0:1] +
        c1 * oh[..., 1:2] +
        c2 * oh[..., 2:3] +
        c3 * oh[..., 3:4] +
        c4 * oh[..., 4:5] +
        c5 * oh[..., 5:6]
    )
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return out

# -----------------------------
# Renderer
# -----------------------------
class VibeRenderer(nn.Module):
    def __init__(self, obj_path: str, device=device):
        super().__init__()
        self.device = device
        verts, faces, _ = load_obj(obj_path, device=device)
        verts = (verts - verts.mean(0)) / (verts.abs().max() + 1e-6)

        self._faces = faces.verts_idx
        self._base_vc = torch.ones_like(verts)[None] * 0.7

        self.vibe_params = AdvancedVibeParams(device=device)

        self.raster_settings = RasterizationSettings(
            image_size=IMAGE_SIZE, blur_radius=0.0, faces_per_pixel=1, bin_size=0
        )

        self.mesh = Meshes(
            verts=[verts],
            faces=[self._faces],
            textures=TexturesVertex(verts_features=self._base_vc.clone()),
        ).to(device)

        R, T = look_at_view_transform(dist=2.5, elev=20, azim=45)
        cams = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60.0)

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cams, raster_settings=self.raster_settings),
            shader=SoftPhongShader(device=device, cameras=cams),
        )

        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).view(1,1,3,3)
        self.register_buffer("_sobel_x", kx)
        self.register_buffer("_sobel_y", ky)

    def _build_cameras(self):
        v = self.vibe_params
        R, T = look_at_view_transform(dist=v.camera_distance.view(1), elev=v.camera_elev.view(1), azim=v.camera_azim.view(1))
        return FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=v.fov.view(1))

    def forward(self, return_mask: bool = True):
        v = self.vibe_params

        cams = self._build_cameras()
        self.renderer.rasterizer.cameras = cams
        self.renderer.shader.cameras = cams

        vc = self._base_vc * v.diffuse_tint.view(1, 1, 3)
        self.mesh.textures = TexturesVertex(verts_features=vc)

        ld = v.light_dir
        ld = ld / (ld.norm() + 1e-8)
        light_pos = (-ld * 3.0).view(1, 3)

        lc = torch.clamp(v.light_color, 0.0, 1.5).view(1, 3)
        li = torch.clamp(v.light_intensity, 0.0, 5.0).view(1, 1)
        diffuse_color = lc * li
        amb = torch.clamp(v.ambient_color, 0.0, 1.5).view(1, 3)

        lights = PointLights(
            device=self.device,
            location=light_pos,
            ambient_color=amb,
            diffuse_color=diffuse_color,
            specular_color=torch.ones((1,3), device=self.device),
        )

        spc = torch.clamp(v.specular_color, 0.0, 1.5)
        sps = torch.clamp(v.specular_strength, 0.0, 1.0)
        rough = torch.clamp(v.roughness, 0.0, 1.0)
        spec_color = (spc * sps * (1.0 - 0.8 * rough)).view(1, 1, 3)

        shin = torch.clamp(v.shininess, 1.0, 128.0)
        shin_eff = torch.clamp(shin * (1.0 - 0.5 * rough) + 1.0, 1.0, 128.0).view(1)

        materials = Materials(device=self.device, specular_color=spec_color, shininess=shin_eff)

        fragments = self.renderer.rasterizer(self.mesh)
        images = self.renderer.shader(fragments, self.mesh, lights=lights, materials=materials)

        rgb = images[0, ..., :3]
        depth = fragments.zbuf[0, ..., 0]

        soft_mask = torch.sigmoid(80.0 * (depth - 1e-6)).unsqueeze(-1)
        depth_norm = torch.clamp((depth - 1.0) / 4.0, 0.0, 1.0)

        H, W = rgb.shape[:2]
        yy = torch.linspace(0, 1, H, device=rgb.device).view(H, 1).repeat(1, W)

        fog_density = torch.clamp(v.fog_density, 0.0, 1.0)
        fog_falloff = torch.clamp(v.fog_falloff, 0.1, 5.0)
        fog_height_bias = torch.clamp(v.fog_height_bias, -1.0, 1.0)

        fog_base = 1.0 - torch.exp(-depth_norm * fog_density * 10.0)
        height_term = torch.sigmoid((yy - (0.5 + 0.4 * fog_height_bias)) * fog_falloff * 6.0)
        fog_f = (torch.clamp(fog_base, 0.0, 1.0) * height_term).unsqueeze(-1)

        fog_color = torch.clamp(v.fog_color, 0.0, 1.5)
        rgb = (1.0 - fog_f) * rgb + fog_f * fog_color
        rgb = rgb * soft_mask + fog_color * (1.0 - soft_mask)

        # Rim
        d = depth_norm * soft_mask[..., 0]
        d4 = d.view(1, 1, H, W)
        gx = F.conv2d(d4, self._sobel_x, padding=1)
        gy = F.conv2d(d4, self._sobel_y, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy).clamp(0.0, 1.0).view(H, W, 1)

        rim_i = torch.clamp(v.rim_intensity, 0.0, 3.0)
        rim_color = torch.clamp(v.rim_color, 0.0, 1.5)
        rgb = torch.clamp(rgb + edge * rim_i * rim_color, 0.0, 1.0)

        # Post
        exposure = torch.clamp(v.exposure, -2.0, 2.0)
        rgb = rgb * (2.0 ** exposure)

        contrast = torch.clamp(v.contrast, 0.5, 2.0)
        rgb = (rgb - 0.5) * contrast + 0.5

        sat = torch.clamp(v.saturation, 0.0, 2.0)
        hue = torch.clamp(v.hue_shift, -0.25, 0.25)
        hsv0 = rgb_to_hsv(torch.clamp(rgb, 0.0, 1.0))
        h = (hsv0[..., 0] + hue) % 1.0
        s = torch.clamp(hsv0[..., 1] * sat, 0.0, 1.0)
        vv = hsv0[..., 2]
        rgb = hsv_to_rgb(torch.stack([h, s, vv], dim=-1))

        gamma = torch.clamp(v.gamma, 0.5, 2.5)
        rgb = torch.pow(torch.clamp(rgb, 0.0, 1.0), 1.0 / gamma)

        vig = torch.clamp(v.vignette_strength, 0.0, 1.0)
        yy2 = torch.linspace(-1, 1, H, device=rgb.device).view(H, 1).repeat(1, W)
        xx2 = torch.linspace(-1, 1, W, device=rgb.device).view(1, W).repeat(H, 1)
        rr = torch.sqrt(xx2 * xx2 + yy2 * yy2)
        vign_mask = torch.clamp(1.0 - vig * (rr ** 1.5), 0.0, 1.0).unsqueeze(-1)
        rgb = rgb * vign_mask

        out = torch.clamp(rgb, 0.0, 1.0)
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

        if return_mask:
            return out, soft_mask
        return out

# -----------------------------
# CLIP losses (FORCE float32)
# -----------------------------
class ClipLosses:
    def __init__(self, device=device):
        self.device = device
        model, _ = clip.load("ViT-B/32", device=device)
        self.model = model.float()  # <- force fp32
        self.model.eval()
        self.mean = torch.tensor([0.481, 0.457, 0.408], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.268, 0.261, 0.275], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def text_feat(self, text: str) -> torch.Tensor:
        text = text.strip()
        if text in self._cache:
            return self._cache[text]
        tokens = clip.tokenize([text]).to(self.device)
        feats = self.model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        self._cache[text] = feats
        return feats

    def img_feat(self, rgb_img: torch.Tensor) -> torch.Tensor:
        x = rgb_img.permute(2, 0, 1).unsqueeze(0)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).float()
        x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feats = self.model.encode_image(x).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def cos_sim(self, rgb_img: torch.Tensor, text: str) -> torch.Tensor:
        tf = self.text_feat(text)
        imf = self.img_feat(rgb_img)
        return (imf * tf).sum()

    def pos_loss(self, rgb_img: torch.Tensor, text: str) -> torch.Tensor:
        return 1.0 - self.cos_sim(rgb_img, text)

    def neg_hinge(self, rgb_img: torch.Tensor, text: str, margin: float) -> torch.Tensor:
        sim = self.cos_sim(rgb_img, text)
        return F.relu(sim - margin)

def masked_compose(rgb: torch.Tensor, mask: torch.Tensor, bg_val: float = 0.0):
    bg = torch.full_like(rgb, bg_val)
    fg_img = rgb * mask + bg * (1.0 - mask)
    bg_img = rgb * (1.0 - mask) + bg * mask
    return fg_img, bg_img

def masked_mean(rgb: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    w = mask
    num = (rgb * w).sum(dim=(0, 1))
    den = w.sum(dim=(0, 1)).clamp_min(eps)
    return num / den

# -----------------------------
# SceneParser (objects + attributes + augmented prompts)
# -----------------------------
@dataclass
class SceneParserConfig:
    model: str = "gpt-4o"
    jpeg_quality: int = 85

class SceneParser:
    def __init__(self, cfg: SceneParserConfig, api_key: str):
        self.cfg = cfg
        self.client = OpenAI(api_key=api_key)

    def _img_url(self, img: torch.Tensor) -> str:
        img_np = (torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
                  .detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        pil = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.cfg.jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _schema(self) -> JsonDict:
        # objects list + attributes (region-aware) + keep your augmented_prompts fallback
        return {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "region": {"type": "string", "enum": ["fg", "bg"]},
                            "attributes": {
                                "type": "object",
                                "properties": {
                                    "color": {"type": "array", "items": {"type": "string"}},
                                    "material": {"type": "array", "items": {"type": "string"}},
                                    "lighting": {"type": "array", "items": {"type": "string"}},
                                    "style": {"type": "array", "items": {"type": "string"}},
                                    "mood": {"type": "array", "items": {"type": "string"}},
                                    "shape": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["color", "material", "lighting", "style", "mood", "shape"],
                                "additionalProperties": False,
                            },
                            "confidence": {"type": "number"}
                        },
                        "required": ["name", "region", "attributes", "confidence"],
                        "additionalProperties": False,
                    }
                },
                "augmented_prompts": {
                    "type": "object",
                    "properties": {
                        "full": {"type": "string"},
                        "fg": {"type": "string"},
                        "bg": {"type": "string"},
                        "neg_fg": {"type": "string"},
                        "neg_bg": {"type": "string"},
                    },
                    "required": ["full", "fg", "bg", "neg_fg", "neg_bg"],
                    "additionalProperties": False,
                },
            },
            "required": ["objects", "augmented_prompts"],
            "additionalProperties": False,
        }

    def parse(self, img: torch.Tensor, prompt: str) -> JsonDict:
        data_url = self._img_url(img)

        system = (
            "Return ONLY JSON.\n"
            "Task:\n"
            "1) Identify the key objects in the image and assign each to region fg or bg.\n"
            "2) For each object, fill attributes: color/material/lighting/style/mood/shape.\n"
            "3) IMPORTANT: If the input prompt contains abstract words (e.g., cozy, cinematic, dreamy),\n"
            "   translate them into concrete, visually-actionable attributes (e.g., warm soft lighting, low contrast,\n"
            "   slightly desaturated, vignette, soft shadows, etc.) using BOTH the prompt and the image context.\n"
            "4) Also provide augmented_prompts(full/fg/bg/neg_fg/neg_bg) that prevent FG/BG attribute swapping.\n"
            "   - fg: foreground object only\n"
            "   - bg: background only\n"
            "   - neg_fg: forbid bg attributes leaking into fg\n"
            "   - neg_bg: forbid fg attributes leaking into bg\n"
            "Keep prompts short, specific, and swap-safe.\n"
        )

        user = f"Prompt:\n{prompt}\n"
        resp = self.client.responses.create(
            model=self.cfg.model,
            instructions=system,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            text={"format": {"type": "json_schema", "name": "scene", "strict": True, "schema": self._schema()}},
        )
        return json.loads(resp.output_text)

MOOD_FALLBACK = {
    "cozy": {
        "lighting": ["warm soft light", "soft shadows"],
        "style": ["gentle vignette", "low contrast"],
        "color": ["warm tones"],
    },
    "cinematic": {
        "lighting": ["dramatic lighting", "strong key light"],
        "style": ["high contrast", "film look"],
        "color": ["slightly desaturated"],
    },
    "dreamy": {
        "lighting": ["soft light", "glow"],
        "style": ["low contrast", "soft vignette"],
        "color": ["pastel tones"],
    },
    "minimal": {
        "style": ["clean", "simple", "uniform background"],
    },
}

def _norm_list(xs):
    xs = xs or []
    out = []
    for x in xs:
        x = str(x).strip()
        if not x:
            continue
        if x.lower() not in [o.lower() for o in out]:
            out.append(x)
    return out

def _merge_attrs(base: dict, extra: dict):
    for k, v in (extra or {}).items():
        base.setdefault(k, [])
        base[k] = _norm_list(list(base[k]) + list(v))
    return base

def _attrs_to_phrase(name: str, attrs: dict, max_items_per_field=3) -> str:
    # Keep it short: pick top few in each attribute field
    color = _norm_list(attrs.get("color", []))[:max_items_per_field]
    material = _norm_list(attrs.get("material", []))[:max_items_per_field]
    lighting = _norm_list(attrs.get("lighting", []))[:max_items_per_field]
    style = _norm_list(attrs.get("style", []))[:max_items_per_field]
    mood = _norm_list(attrs.get("mood", []))[:max_items_per_field]
    shape = _norm_list(attrs.get("shape", []))[:max_items_per_field]

    bits = []
    if color: bits.append(", ".join(color))
    if material: bits.append(", ".join(material))
    if lighting: bits.append(", ".join(lighting))
    if style: bits.append(", ".join(style))
    if mood: bits.append(", ".join(mood))
    if shape: bits.append(", ".join(shape))

    desc = ", ".join(bits).strip()
    if desc:
        return f"a {name}, {desc}"
    return f"a {name}"

def _extract_scene_objects(scene: Optional[JsonDict]) -> Tuple[List[dict], List[dict]]:
    fg_objs, bg_objs = [], []
    if not scene or "objects" not in scene:
        return fg_objs, bg_objs
    for o in scene.get("objects", []):
        if not isinstance(o, dict):
            continue
        reg = o.get("region", "")
        if reg == "fg":
            fg_objs.append(o)
        elif reg == "bg":
            bg_objs.append(o)
    return fg_objs, bg_objs

def _collect_attrs(objs: List[dict]) -> dict:
    merged = {"color": [], "material": [], "lighting": [], "style": [], "mood": [], "shape": []}
    for o in objs:
        attrs = (o or {}).get("attributes", {}) if isinstance(o, dict) else {}
        _merge_attrs(merged, attrs)
    return merged

def build_prompts_v2(scene: Optional[JsonDict], raw_prompt: str) -> Tuple[str, str, str, str, str]:
    """
    Returns: full_prompt, fg_prompt, bg_prompt, neg_fg, neg_bg
    Preference order:
      1) objects+attributes -> build our own swap-safe prompts
      2) fallback to scene["augmented_prompts"]
      3) fallback to raw_prompt
    """
    fg_objs, bg_objs = _extract_scene_objects(scene)
    raw_lower = (raw_prompt or "").lower()

    # If VLM didn't identify bg object, treat as "background"
    if not bg_objs:
        bg_objs = [{"name": "background", "region": "bg",
                    "attributes": {"color": [], "material": [], "lighting": [], "style": [], "mood": [], "shape": []},
                    "confidence": 0.2}]

    # Fallback mapping for abstract mood words
    for k, add in MOOD_FALLBACK.items():
        if k in raw_lower:
            # Apply to bg by default; you can change if you want
            bg_objs[0]["attributes"] = _merge_attrs(bg_objs[0].get("attributes", {}), add)

    # Pick a main FG object name
    fg_name = fg_objs[0].get("name", "object") if fg_objs else "object"
    bg_name = "background"

    fg_attrs = _collect_attrs(fg_objs)
    bg_attrs = _collect_attrs(bg_objs)

    fg_prompt = _attrs_to_phrase(fg_name, fg_attrs)
    bg_prompt = _attrs_to_phrase(bg_name, bg_attrs)

    # Make bg prompt explicitly "background only"
    bg_prompt = bg_prompt + ", uniform backdrop, no foreground object"

    # Swap-guard negatives:
    # - neg_fg: forbid bg colors/styles leaking into fg
    # - neg_bg: forbid fg colors/object names leaking into bg
    bg_colors = _norm_list(bg_attrs.get("color", []))
    fg_colors = _norm_list(fg_attrs.get("color", []))

    neg_fg_bits = []
    if bg_colors:
        neg_fg_bits += [f"not {c}" for c in bg_colors[:4]]
    neg_fg_bits += ["not background", "not backdrop"]

    neg_bg_bits = []
    if fg_colors:
        neg_bg_bits += [f"no {c}" for c in fg_colors[:4]]
    # also forbid the fg object name in bg
    if fg_name and fg_name != "object":
        neg_bg_bits += [f"no {fg_name}", "no animal", "no subject"]

    neg_fg = ", ".join(neg_fg_bits).strip()
    neg_bg = ", ".join(neg_bg_bits).strip()

    full_prompt = raw_prompt.strip() if raw_prompt else f"{fg_prompt} on {bg_prompt}"

    # If we had nothing meaningful, fallback to scene's augmented prompts
    if (not fg_objs and scene and isinstance(scene.get("augmented_prompts", None), dict)):
        ap = scene["augmented_prompts"]
        return ap["full"], ap["fg"], ap["bg"], ap["neg_fg"], ap["neg_bg"]

    return full_prompt, fg_prompt, bg_prompt, neg_fg, neg_bg

@dataclass
class ViberConfinerConfig:
    model: str = "gpt-4o"
    jpeg_quality: int = 85

class ViberConfiner:
    def __init__(self, cfg: ViberConfinerConfig, api_key: str):
        self.cfg = cfg
        self.client = OpenAI(api_key=api_key)

        self.param_names = [
            "fog_density","fog_color","fog_falloff","fog_height_bias",
            "light_intensity","light_color","light_dir","rim_intensity","rim_color","ambient_color",
            "camera_distance","camera_elev","camera_azim","fov",
            "diffuse_tint","specular_strength","specular_color","roughness","shininess",
            "contrast","exposure","gamma","saturation","hue_shift","vignette_strength",
        ]

    def _img_url(self, img: torch.Tensor) -> str:
        img_np = (torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        pil = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.cfg.jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _catalog(self) -> JsonDict:
        return {
            "fog_density": {"range":[0.0,1.0]},
            "fog_color": {"range":[[0.0,0.0,0.0],[1.5,1.5,1.5]]},
            "fog_falloff": {"range":[0.1,5.0]},
            "fog_height_bias": {"range":[-1.0,1.0]},

            "light_intensity":{"range":[0.0,5.0]},
            "light_color":{"range":[[0.0,0.0,0.0],[1.5,1.5,1.5]]},
            "light_dir":{"range":[[-1.0,-1.0,-1.0],[1.0,1.0,1.0]]},
            "rim_intensity":{"range":[0.0,3.0]},
            "rim_color":{"range":[[0.0,0.0,0.0],[1.5,1.5,1.5]]},
            "ambient_color":{"range":[[0.0,0.0,0.0],[1.5,1.5,1.5]]},

            "camera_distance":{"range":[1.0,6.0]},
            "camera_elev":{"range":[-80.0,80.0]},
            "camera_azim":{"range":[-180.0,180.0]},
            "fov":{"range":[10.0,120.0]},

            "diffuse_tint":{"range":[[0.0,0.0,0.0],[2.0,2.0,2.0]]},
            "specular_strength":{"range":[0.0,1.0]},
            "specular_color":{"range":[[0.0,0.0,0.0],[1.5,1.5,1.5]]},
            "roughness":{"range":[0.0,1.0]},
            "shininess":{"range":[1.0,128.0]},

            "contrast":{"range":[0.5,2.0]},
            "exposure":{"range":[-2.0,2.0]},
            "gamma":{"range":[0.5,2.5]},
            "saturation":{"range":[0.0,2.0]},
            "hue_shift":{"range":[-0.25,0.25]},
            "vignette_strength":{"range":[0.0,1.0]},
        }

    def _schema(self) -> JsonDict:
        param_item = {
            "type":"object",
            "properties":{
                "optimize":{"type":"boolean"},
                "range":{
                    "anyOf":[
                        {"type":"array","minItems":2,"maxItems":2,"items":{"type":"number"}},
                        {"type":"array","minItems":2,"maxItems":2,
                         "items":{"type":"array","minItems":3,"maxItems":3,"items":{"type":"number"}}},
                    ]
                },
                "reason":{"type":"string"},
                "priority":{"type":"integer"},
            },
            "required":["optimize","range","reason","priority"],
            "additionalProperties":False,
        }
        return {
            "type":"object",
            "properties":{n:param_item for n in self.param_names},
            "required":self.param_names,
            "additionalProperties":False,
        }

    def query(
        self,
        img: torch.Tensor,
        prompt: str,
        fg: str,
        bg: str,
        cur: JsonDict,
        bg_hint: Optional[List[float]],
    ):
        def _clamp_range_to_base(rng, base):
            """
            Decide scalar/vec3 by BASE shape (source of truth).
            If rng shape mismatches, fallback to base (stable).
            """
            base_is_vec3 = (
                isinstance(base, list) and len(base) == 2 and
                isinstance(base[0], list) and isinstance(base[1], list) and
                len(base[0]) == 3 and len(base[1]) == 3
            )
            rng_is_scalar = (
                isinstance(rng, list) and len(rng) == 2 and
                all(isinstance(x, (int, float)) for x in rng)
            )
            rng_is_vec3 = (
                isinstance(rng, list) and len(rng) == 2 and
                isinstance(rng[0], list) and isinstance(rng[1], list) and
                len(rng[0]) == 3 and len(rng[1]) == 3 and
                all(isinstance(x, (int, float)) for x in (rng[0] + rng[1]))
            )

            # Base expects vec3
            if base_is_vec3:
                if not rng_is_vec3:
                    return base
                blo, bhi = base
                lo = [max(float(min(a, b)), float(c)) for a, b, c in zip(rng[0], rng[1], blo)]
                hi = [min(float(max(a, b)), float(c)) for a, b, c in zip(rng[0], rng[1], bhi)]
                for i in range(3):
                    if hi[i] < lo[i]:
                        lo[i], hi[i] = float(blo[i]), float(bhi[i])
                return [lo, hi]

            # Base expects scalar
            else:
                if not rng_is_scalar:
                    return base
                lo, hi = float(min(rng)), float(max(rng))
                lo = max(lo, float(base[0]))
                hi = min(hi, float(base[1]))
                if hi < lo:
                    return base
                return [lo, hi]

        data_url = self._img_url(img)
        catalog = self._catalog()

        hint = f"BG color hint RGB approx: {bg_hint}\n" if bg_hint else ""
        system = (
            "Return ONLY JSON that matches schema.\n"
            "Decide optimize=true/false for each parameter.\n"
            "Keep ranges within the provided catalog bounds.\n"
            "Use priority=1..K for optimized params (1 is most important), and priority=0 for optimize=false.\n"
            "Avoid attribute swapping between FG/BG.\n"
        )
        user = (
            f"Prompt: {prompt}\n"
            f"FG prompt: {fg}\n"
            f"BG prompt: {bg}\n"
            f"{hint}"
            f"Catalog: {json.dumps(catalog)}\n"
            f"Current: {json.dumps(cur)}\n"
            "Return plan."
        )

        resp = self.client.responses.create(
            model=self.cfg.model,
            instructions=system,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            text={"format": {"type": "json_schema", "name": "plan", "strict": True, "schema": self._schema()}},
        )
        plan = json.loads(resp.output_text)

        # ---- sanitize (shape-aware clamp + keep optimize) ----
        clean: JsonDict = {}
        for name, spec in catalog.items():
            cfg = plan.get(name, {})
            base = spec["range"]
            rng = cfg.get("range", base)
            rng = _clamp_range_to_base(rng, base)

            opt = bool(cfg.get("optimize", False))
            reason = str(cfg.get("reason", "")).strip() or ("selected" if opt else "not selected")

            prio = cfg.get("priority", 0)
            try:
                prio = int(prio)
            except Exception:
                prio = 0
            if not opt:
                prio = 0
            elif prio <= 0:
                prio = 1  # if optimized but bad priority, repair

            clean[name] = {
                "optimize": opt,
                "range": rng,
                "reason": reason,
                "priority": prio,
            }

        return clean

def build_prompts(scene: Optional[JsonDict], target_prompt: str):
    if scene and "augmented_prompts" in scene:
        ap = scene["augmented_prompts"]
        return ap["full"], ap["fg"], ap["bg"], ap["neg_fg"], ap["neg_bg"]
    # fallback
    return target_prompt, target_prompt, target_prompt, "", ""

def apply_plan(params: AdvancedVibeParams, plan: JsonDict):
    # 1) update ranges (for ALL params, even if not optimizing)
    for name, cfg in plan.items():
        if hasattr(params, name) and isinstance(getattr(params, name), nn.Parameter):
            params.update_range(name, cfg["range"])

    # 2) choose top-K by priority among optimize=True
    candidates = []
    for name, cfg in plan.items():
        if not (hasattr(params, name) and isinstance(getattr(params, name), nn.Parameter)):
            continue
        if not isinstance(cfg, dict):
            continue
        prio = cfg.get("priority", 10**9)
        try:
            prio = int(prio)
        except Exception:
            prio = 10**9
        candidates.append((prio, name))

    candidates.sort(key=lambda x: (x[0], x[1]))  # priority asc, then name
    selected = set([name for _, name in candidates[:MAX_OPT_NUM]])

    # 3) set requires_grad based on selection
    active = []
    for name, p in params.named_parameters():
        opt = (name in selected)
        p.requires_grad_(opt)
        if opt:
            active.append(p)

    # 4) hard fallback: never allow empty active set
    if not active:
        for p in params.parameters():
            p.requires_grad_(True)
        active = list(params.parameters())

    return active

def crop_penalty(mask, border=12):
    # mask: (H,W,1) in [0,1]
    m = mask[..., 0]
    top = m[:border, :].mean()
    bot = m[-border:, :].mean()
    lef = m[:, :border].mean()
    rig = m[:, -border:].mean()
    return top + bot + lef + rig  # 越大代表越貼邊/被裁切風險越高

# -----------------------------
# Run
# -----------------------------
def run_closed_loop(obj_path: str, prompt: str, out_image_path: Optional[str] = None):
    print("device:", device)
    renderer = VibeRenderer(obj_path, device=device).to(device)
    clip_losses = ClipLosses(device=device)

    parser = SceneParser(SceneParserConfig(), OPENAI_API_KEY)
    confiner = ViberConfiner(ViberConfinerConfig(), OPENAI_API_KEY)

    # initial parse
    with torch.no_grad():
        img0, _ = renderer(return_mask=True)
    scene = None
    try:
        scene = parser.parse(img0, prompt)
    except Exception as e:
        print("[warn] SceneParser failed:", e)

    full_p, fg_p, bg_p, neg_fg, neg_bg = build_prompts_v2(scene, prompt)

    # force stronger swap-prevent negatives if prompt implies bg color
    bg_col = get_bg_color_target(prompt)
    if bg_col is not None:
        # mechanical negatives (strong!)
        if not neg_fg.strip():
            neg_fg = "a navy blue horse, a blue horse, a dark blue horse"
        if not neg_bg.strip():
            neg_bg = "a pink background, a magenta background, a red background"

    for outer in range(STEPS_OUTER):
        with torch.no_grad():
            img, mask = renderer(return_mask=True)
            cur = renderer.vibe_params.get_state_values()

        bg_hint = bg_col.detach().cpu().tolist() if bg_col is not None else None
        plan = confiner.query(img, prompt, fg_p, bg_p, cur, bg_hint=bg_hint)

        # log top
        chosen = [(n,c) for n,c in plan.items() if c.get("optimize")]
        chosen = sorted(chosen, key=lambda x: x[1].get("priority", 10**9))
        print(f"\n[outer {outer}] Confiner plan (optimize=True): {len(chosen)} params")
        for n, c in chosen[:12]:
            print(f"  - {n:18s} prio={c.get('priority')} range={c.get('range')} | {c.get('reason')}")
        
        active = apply_plan(renderer.vibe_params, plan)
        optim = torch.optim.Adam(active, lr=LR)

        for it in range(STEPS_INNER):
            optim.zero_grad(set_to_none=True)
            rgb, m = renderer(return_mask=True)

            fg_img, bg_img = masked_compose(rgb, m, bg_val=0.0)

            L_fg = clip_losses.pos_loss(fg_img, fg_p)
            L_bg = clip_losses.pos_loss(bg_img, bg_p)
            L_full = clip_losses.pos_loss(rgb, full_p)

            L_neg = torch.tensor(0.0, device=device)
            if neg_fg.strip():
                L_neg = L_neg + clip_losses.neg_hinge(fg_img, neg_fg, margin=NEG_MARGIN)
            if neg_bg.strip():
                L_neg = L_neg + clip_losses.neg_hinge(bg_img, neg_bg, margin=NEG_MARGIN)

            L_bgcol = torch.tensor(0.0, device=device)
            if bg_col is not None:
                bg_mean = masked_mean(rgb, (1.0 - m))
                L_bgcol = F.mse_loss(bg_mean, bg_col)

            loss = (W_FG*L_fg) + (W_BG*L_bg) + (W_FULL*L_full) + (W_NEG*L_neg) + (W_BG_COLOR*L_bgcol)

            L_crop = crop_penalty(m, border=12)
            loss = loss + 1.0 * L_crop

            # if anything becomes nan, stop early and repair
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at outer={outer} it={it}, repairing params...")
                with torch.no_grad():
                    for p in renderer.vibe_params.parameters():
                        safe_nan_to_num_(p, nan=0.0, posinf=1.0, neginf=0.0)
                    renderer.vibe_params.clamp_()
                break

            loss.backward()

            # grad clip (key)
            torch.nn.utils.clip_grad_norm_(active, GRAD_CLIP)

            optim.step()

            # clamp + nan repair (key)
            renderer.vibe_params.clamp_()

        with torch.no_grad():
            out, _ = renderer(return_mask=True)
            params = renderer.vibe_params.get_state_values()
            print(f"[outer {outer}] loss={float(loss.item()):.4f} | "
                  f"L_fg={float(L_fg.item()):.3f} L_bg={float(L_bg.item()):.3f} "
                  f"L_full={float(L_full.item()):.3f} L_neg={float(L_neg.item()):.3f} "
                  f"L_bgcol={float(L_bgcol.item()):.3f}")
            show_progress(out, params, title=f"outer={outer}_loss={float(loss.item()):.4f}")
            if out_image_path:
                save_render(out, out_image_path)
    return renderer

if __name__ == "__main__":
    obj = os.getenv("VIBE_OBJ_PATH", os.getenv("VIBE_OBJ", "data/mesh_upright.obj"))
    out_img = os.getenv("VIBE_OUT_IMAGE", "render.png")
    prompt = os.getenv(
        "VIBE_PROMPT",
        "Bright cyan ceramic horse (glossy, low roughness, sharp specular highlights), pure mustard-yellow studio background. Neutral white key light from upper-left, subtle white rim light. Medium contrast, slight vignette, no fog."
    )
    run_closed_loop(obj, prompt, out_image_path=out_img)