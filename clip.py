# ============================================================
# Baseline Vibe Optimization (CLIP only) — Naive E2E
# - NO VLM / NO API calls
# - NO text parsing / augmentation
# - Optimize ALL params end-to-end with CLIP loss on full image only
# - Keeps: CLIP float32, grad clipping, nan_to_num guards, clamp_
# - Saves rendered result to VIBE_OUT_IMAGE
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
STEPS = int(os.getenv("VIBE_STEPS", "240"))  # single loop baseline
LR = float(os.getenv("VIBE_LR", "0.02"))

GRAD_CLIP = float(os.getenv("GRAD_CLIP", "1.0"))
W_FULL = float(os.getenv("W_FULL", "1.0"))

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
    if torch.is_floating_point(t):
        torch.nan_to_num_(t, nan=nan, posinf=posinf, neginf=neginf)

def show_progress(img_t: torch.Tensor, params: dict, title: str = ""):
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
            for _, p in self.named_parameters():
                safe_nan_to_num_(p, nan=0.0, posinf=1.0, neginf=0.0)

            for name, p in self.named_parameters():
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

    def forward(self):
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

        d = depth_norm * soft_mask[..., 0]
        d4 = d.view(1, 1, H, W)
        gx = F.conv2d(d4, self._sobel_x, padding=1)
        gy = F.conv2d(d4, self._sobel_y, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy).clamp(0.0, 1.0).view(H, W, 1)

        rim_i = torch.clamp(v.rim_intensity, 0.0, 3.0)
        rim_color = torch.clamp(v.rim_color, 0.0, 1.5)
        rgb = torch.clamp(rgb + edge * rim_i * rim_color, 0.0, 1.0)

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
        return out

# -----------------------------
# CLIP losses (FORCE float32)
# -----------------------------
class ClipLosses:
    def __init__(self, device=device):
        self.device = device
        model, _ = clip.load("ViT-B/32", device=device)
        self.model = model.float()
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

# -----------------------------
# Run (naive CLIP-only)
# -----------------------------
def run_clip_only(obj_path: str, prompt: str, out_image_path: Optional[str] = None):
    print("device:", device)
    renderer = VibeRenderer(obj_path, device=device).to(device)
    clip_losses = ClipLosses(device=device)

    # (A) freeze camera
    freeze = {"camera_distance", "camera_elev", "camera_azim", "fov"}

    # (B) disable fog (keep it constant)
    freeze |= {"fog_density", "fog_color", "fog_falloff", "fog_height_bias"}

    trainable = []
    for name, p in renderer.vibe_params.named_parameters():
        p.requires_grad_(name not in freeze)
        if p.requires_grad:
            trainable.append(p)

    # hard set fog off once
    with torch.no_grad():
        renderer.vibe_params.fog_density.fill_(0.0)

    optim = torch.optim.Adam(trainable, lr=LR)

    loss = torch.tensor(0.0, device=device)
    for it in range(STEPS):
        optim.zero_grad(set_to_none=True)
        rgb = renderer()

        loss = W_FULL * clip_losses.pos_loss(rgb, prompt)

        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at it={it}, repairing params...")
            with torch.no_grad():
                for p in renderer.vibe_params.parameters():
                    safe_nan_to_num_(p, nan=0.0, posinf=1.0, neginf=0.0)
                renderer.vibe_params.clamp_()
                renderer.vibe_params.fog_density.fill_(0.0)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optim.step()
        renderer.vibe_params.clamp_()

        # keep fog off
        with torch.no_grad():
            renderer.vibe_params.fog_density.fill_(0.0)

        if (it + 1) % 40 == 0:
            with torch.no_grad():
                print(f"[it {it+1}/{STEPS}] loss={float(loss.item()):.4f}")
                show_progress(rgb, renderer.vibe_params.get_state_values(), title=f"it={it+1}_loss={float(loss.item()):.4f}")
                if out_image_path:
                    save_render(rgb, out_image_path)

    with torch.no_grad():
        out = renderer()
        if out_image_path:
            save_render(out, out_image_path)
    return renderer

if __name__ == "__main__":
    obj = os.getenv("VIBE_OBJ_PATH", os.getenv("VIBE_OBJ", "data/mesh_upright.obj"))
    out_img = os.getenv("VIBE_OUT_IMAGE", "render_baseline.png")
    prompt = os.getenv("VIBE_PROMPT", "A shiny object on a plain background.")
    run_clip_only(obj, prompt, out_image_path=out_img)
