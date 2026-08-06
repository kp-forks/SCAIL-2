#!/usr/bin/env python3
"""Cache SCAIL-2 training examples into latent WebDataset shards.

Each input case follows the same file convention as inference examples, with one
extra target video:

    case_dir/
      GT.mp4                  # training target video
      rendered_v2.mp4         # driving / pose video, preferred
      driving.mp4             # fallback driving / pose video
      rendered_mask_v2.mp4    # driving mask, preferred
      replace_mask.mp4        # fallback driving mask
      ref.jpg|ref.png
      ref_mask.jpg|ref_mask.png
      prompt.txt|text.txt     # optional

The output fields intentionally match the latent reader used by SCAIL training:
video_pth.zstd is GT, smpl_render_downsample.zstd is driving/pose latent,
latent_sam_mask.zstd is driving mask, latent_ref_mask.zstd is reference mask.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import zstandard as zstd
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from wan.modules.vae import WanVAE
from wan.utils.scail_utils import (
    extract_and_compress_mask_to_latent,
    load_image_to_tensor_chw_normalized,
    load_video_for_pose_sample,
    resize_for_rectangle_crop,
)


VIDEO_PATTERNS = ("GT.mp4", "gt.mp4")
POSE_PATTERNS = ("rendered_v2.mp4", "driving.mp4", "rendered.mp4", "smpl_render.mp4")
DRIVING_MASK_PATTERNS = ("rendered_mask_v2.mp4", "replace_mask.mp4", "mask_video.mp4")
REF_PATTERNS = ("ref.jpg", "ref.png", "ref_image.jpg", "ref_image.png", "ref_image_with_mask.png")
REF_MASK_PATTERNS = ("ref_mask.jpg", "ref_mask.png", "mask_image.jpg", "mask_image.png")
PROMPT_PATTERNS = ("prompt.txt", "text.txt", "caption.txt", "recaption.txt")


@dataclass
class CasePaths:
    key: str
    case_dir: Path
    gt: Path
    pose: Path
    driving_mask: Path
    ref: Path
    ref_mask: Path
    prompt: str


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def first_existing(directory: Path, patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        path = directory / pattern
        if path.is_file():
            return path
    return None


def read_prompt(case_dir: Path, override: str | None) -> str:
    if override is not None:
        return override
    prompt_path = first_existing(case_dir, PROMPT_PATTERNS)
    if prompt_path is None:
        return ""
    return prompt_path.read_text(encoding="utf-8").strip()


def discover_case(case_dir: Path, prompt_override: str | None = None) -> CasePaths:
    case_dir = case_dir.resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Input case is not a directory: {case_dir}")

    gt = first_existing(case_dir, VIDEO_PATTERNS)
    pose = first_existing(case_dir, POSE_PATTERNS)
    driving_mask = first_existing(case_dir, DRIVING_MASK_PATTERNS)
    ref = first_existing(case_dir, REF_PATTERNS)
    ref_mask = first_existing(case_dir, REF_MASK_PATTERNS)

    missing = []
    for name, value, patterns in (
        ("GT video", gt, VIDEO_PATTERNS),
        ("driving/pose video", pose, POSE_PATTERNS),
        ("driving mask video", driving_mask, DRIVING_MASK_PATTERNS),
        ("reference image", ref, REF_PATTERNS),
        ("reference mask image", ref_mask, REF_MASK_PATTERNS),
    ):
        if value is None:
            missing.append(f"{name} ({', '.join(patterns)})")
    if missing:
        raise FileNotFoundError(f"{case_dir} is missing: " + "; ".join(missing))

    return CasePaths(
        key=safe_key(case_dir.name),
        case_dir=case_dir,
        gt=gt,
        pose=pose,
        driving_mask=driving_mask,
        ref=ref,
        ref_mask=ref_mask,
        prompt=read_prompt(case_dir, prompt_override),
    )


def load_cases_from_txt(path: Path) -> list[tuple[Path, str | None]]:
    cases = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "@@" in line:
            prompt, case_dir = line.split("@@", 1)
            cases.append((Path(case_dir.strip()), prompt.strip()))
        else:
            cases.append((Path(line), None))
    if not cases:
        raise ValueError(f"No input cases found in {path}")
    return cases


def safe_key(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    key = "".join(keep).strip("._")
    return key or "sample"


def serialize_tensor_to_zstd(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    buffer.seek(0)
    return zstd.ZstdCompressor(level=22).compress(buffer.read())


def serialize_text(value: str) -> bytes:
    return value.encode("utf-8")


def write_wds_sample(tar: tarfile.TarFile, key: str, sample: dict[str, bytes | str]) -> None:
    for suffix, value in sample.items():
        if suffix == "__key__":
            continue
        data = value if isinstance(value, bytes) else serialize_text(str(value))
        info = tarfile.TarInfo(f"{key}.{suffix}")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def load_video_tchw(path: Path, target_size: tuple[int, int], max_frames: int | None) -> torch.Tensor:
    video = load_video_for_pose_sample(str(path))
    if max_frames is not None:
        video = video[:max_frames]
    video = video.permute(0, 3, 1, 2)
    video = resize_for_rectangle_crop(video, target_size, reshape_mode="center")
    return (video - 127.5) / 127.5


def load_image_chw(path: Path, target_size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = load_image_to_tensor_chw_normalized(image)
    tensor = resize_for_rectangle_crop(tensor, target_size, reshape_mode="center")
    return tensor.squeeze(0)


def trim_to_common_length(*videos: torch.Tensor) -> tuple[torch.Tensor, ...]:
    length = min(video.shape[0] for video in videos)
    keep = ((length - 1) // 4) * 4 + 1
    if keep < 1:
        raise ValueError("Video is too short after VAE temporal alignment.")
    return tuple(video[:keep].contiguous() for video in videos)


def make_input_from_first_frame(first_frame: torch.Tensor, target_video: torch.Tensor) -> torch.Tensor:
    return torch.cat([first_frame, torch.zeros_like(target_video[:, 1:])], dim=1)


def build_ref_mask(ref_mask: torch.Tensor) -> torch.Tensor:
    return extract_and_compress_mask_to_latent(
        ref_mask.unsqueeze(1), additional_spatial_downsample=1
    )


def encode_case(case: CasePaths, vae: WanVAE, args: argparse.Namespace, device: torch.device) -> dict[str, bytes | str]:
    target_size = (args.target_h, args.target_w)

    gt = load_video_tchw(case.gt, target_size, args.max_frames)
    pose = load_video_tchw(case.pose, target_size, args.max_frames)
    driving_mask = load_video_tchw(case.driving_mask, target_size, args.max_frames)
    gt, pose, driving_mask = trim_to_common_length(gt, pose, driving_mask)

    ref = load_image_chw(case.ref, target_size)
    ref_mask = load_image_chw(case.ref_mask, target_size)

    gt_cthw = rearrange(gt, "t c h w -> c t h w").to(device=device, dtype=args.dtype)
    pose_downsample = F.interpolate(
        pose.to(device=device, dtype=args.dtype),
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
    )
    pose_cthw = rearrange(pose_downsample, "t c h w -> c t h w")
    ref_cthw = ref.unsqueeze(1).to(device=device, dtype=args.dtype)

    with torch.no_grad():
        gt_latent = vae.encode([gt_cthw])[0].to(torch.bfloat16).cpu()
        pose_latent = vae.encode([pose_cthw])[0].to(torch.bfloat16).cpu()
        ref_latent = vae.encode([ref_cthw])[0].to(torch.bfloat16).cpu()
        first_frame_latent = vae.encode([
            make_input_from_first_frame(ref_cthw, gt_cthw)
        ])[0].to(torch.bfloat16).cpu()

    driving_mask_downsample = F.interpolate(
        rearrange(driving_mask, "t c h w -> c t h w").float(),
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
    )
    driving_mask_latent = extract_and_compress_mask_to_latent(
        driving_mask_downsample, additional_spatial_downsample=1
    ).to(torch.bfloat16).cpu()

    ref_mask_latent = build_ref_mask(ref_mask.to(dtype=torch.float32)).to(torch.bfloat16).cpu()

    zero_bbox_mask = torch.zeros(
        gt_latent.shape[1],
        gt_latent.shape[0],
        gt_latent.shape[2],
        gt_latent.shape[3],
        dtype=torch.bfloat16,
    )

    sample = {
        "__key__": case.key,
        "video_pth.zstd": serialize_tensor_to_zstd(gt_latent),
        "smpl_render_downsample.zstd": serialize_tensor_to_zstd(pose_latent),
        "smpl_render_aug_downsample.zstd": serialize_tensor_to_zstd(pose_latent),
        "first_frame_pth.zstd": serialize_tensor_to_zstd(first_frame_latent),
        "ref_frame_pth.zstd": serialize_tensor_to_zstd(ref_latent),
        "pixel_first_frame.zstd": serialize_tensor_to_zstd(ref_cthw.to(torch.bfloat16).cpu()),
        "latent_sam_mask.zstd": serialize_tensor_to_zstd(driving_mask_latent),
        "latent_sam_aug_mask.zstd": serialize_tensor_to_zstd(driving_mask_latent),
        "latent_ref_mask.zstd": serialize_tensor_to_zstd(ref_mask_latent),
        "latent_hands_mask.zstd": serialize_tensor_to_zstd(zero_bbox_mask),
        "latent_faces_mask.zstd": serialize_tensor_to_zstd(zero_bbox_mask),
        "latent_ocr_mask.zstd": serialize_tensor_to_zstd(zero_bbox_mask),
        "frame": str(gt_latent.shape[1]),
        "recaption": case.prompt,
        "id": "0",
        "e2e_flag": str(args.e2e_flag),
        "ref_mask_flag": str(args.ref_mask_flag),
        "meta.json": json.dumps(
            {
                "case_dir": str(case.case_dir),
                "gt": str(case.gt),
                "pose": str(case.pose),
                "driving_mask": str(case.driving_mask),
                "ref": str(case.ref),
                "ref_mask": str(case.ref_mask),
                "target_size": [args.target_h, args.target_w],
                "pixel_frames": int(gt.shape[0]),
                "latent_shape": list(gt_latent.shape),
            },
            ensure_ascii=False,
        ),
    }
    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=str, default=None, help="One SCAIL example/case directory.")
    parser.add_argument("--input_list", type=str, default=None, help="Text file: case_dir or prompt@@case_dir per line.")
    parser.add_argument("--output", type=str, required=True, help="Output .tar path.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Wan/SCAIL checkpoint directory containing Wan2.1_VAE.pth.")
    parser.add_argument("--vae_checkpoint", type=str, default="Wan2.1_VAE.pth", help="VAE filename under --ckpt_dir, or an absolute path.")
    parser.add_argument("--target_h", type=int, default=512)
    parser.add_argument("--target_w", type=int, default=896)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="Override prompt when --input_dir is used.")
    parser.add_argument("--e2e_flag", type=str2bool, default=True)
    parser.add_argument("--ref_mask_flag", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    if (args.input_dir is None) == (args.input_list is None):
        raise ValueError("Specify exactly one of --input_dir or --input_list.")
    if not args.output.endswith(".tar"):
        raise ValueError("--output should be a .tar path.")

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    args.dtype = dtype_map[args.dtype]
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    if args.input_dir is not None:
        case_specs = [(Path(args.input_dir), args.prompt)]
    else:
        case_specs = load_cases_from_txt(Path(args.input_list))

    cases = [discover_case(case_dir, prompt_override) for case_dir, prompt_override in case_specs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    vae_path = Path(args.vae_checkpoint)
    if not vae_path.is_absolute():
        vae_path = Path(args.ckpt_dir) / vae_path
    if not vae_path.is_file():
        raise FileNotFoundError(f"VAE checkpoint does not exist: {vae_path}")

    device = torch.device(args.device)
    vae = WanVAE(vae_pth=str(vae_path), dtype=args.dtype, device=device)

    tmp_output = output.with_suffix(output.suffix + ".tmp")
    with tarfile.open(tmp_output, "w") as tar:
        for idx, case in enumerate(tqdm(cases, desc="Caching SCAIL-2 cases")):
            if len(cases) > 1:
                case.key = f"{idx:07d}_{case.key}"
            logging.info("Caching %s", case.case_dir)
            sample = encode_case(case, vae, args, device)
            write_wds_sample(tar, case.key, sample)

    os.replace(tmp_output, output)
    logging.info("Wrote %d sample(s) to %s", len(cases), output)


if __name__ == "__main__":
    main()
