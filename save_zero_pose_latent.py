#!/usr/bin/env python3
"""Generate Wan VAE latents for an empty / black SCAIL pose condition.

The generated tensor is the Wan VAE encoding of a pixel-space pose video filled
with -1, i.e. black in the normalized [-1, 1] range. It is saved in the format
used by SCAIL training pose dropout:

    zero_pose_latent_{T_latent}_{H_latent}_{W_latent}.pt

Each saved tensor has shape [T_latent, 16, H_latent, W_latent]. The image sizes
passed to this script are the actual blank pose sizes fed to VAE. For pipelines
that downsample the driving/pose video by 0.5 before VAE, pass the downsampled
pose size, not the final GT video size.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm

from wan.modules.vae import WanVAE


DEFAULT_LENGTHS = (33, 49, 65, 81)
DEFAULT_IMAGE_SIZES = (
    (256, 448),
    (448, 256),
    (352, 640),
    (640, 352),
    (512, 896),
    (896, 512),
    (704, 1280),
    (1280, 704),
)


def parse_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Checkpoint directory containing Wan2.1_VAE.pth. Not needed if --vae_path is absolute.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="Wan2.1_VAE.pth",
        help="VAE checkpoint path, or filename under --ckpt_dir. Default: Wan2.1_VAE.pth.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./latents",
        help="Directory where zero_pose_latent_*.pt files are written.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_LENGTHS),
        help="Pixel-frame lengths to encode. Default: 33 49 65 81.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        action="append",
        metavar=("H", "W"),
        help="VAE input pose size. Can be repeated. Defaults to common SCAIL sizes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for VAE encoding.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Autocast/model dtype used for VAE encoding.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing zero_pose_latent_*.pt files.",
    )
    args = parser.parse_args()

    vae_path = Path(args.vae_path)
    if not vae_path.is_absolute():
        if args.ckpt_dir is None:
            raise ValueError("--ckpt_dir is required when --vae_path is not absolute.")
        vae_path = Path(args.ckpt_dir) / vae_path
    if not vae_path.is_file():
        raise FileNotFoundError(f"VAE checkpoint does not exist: {vae_path}")
    args.vae_path = vae_path

    args.image_sizes = [tuple(x) for x in (args.image_size or DEFAULT_IMAGE_SIZES)]
    args.dtype = parse_dtype(args.dtype)
    return args


def encode_zero_pose(model: WanVAE, length: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    zero_pose = torch.full(
        size=(3, length, height, width),
        fill_value=-1,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        latent = model.encode([zero_pose])[0]
    return latent.permute(1, 0, 2, 3).contiguous().to(torch.bfloat16).cpu()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logging.info("Loading VAE from %s", args.vae_path)
    model = WanVAE(vae_pth=str(args.vae_path), dtype=args.dtype, device=device)

    jobs = [(length, h, w) for length in args.lengths for h, w in args.image_sizes]
    for length, height, width in tqdm(jobs, desc="Encoding zero pose latents"):
        latent = encode_zero_pose(model, length, height, width, device, args.dtype)
        latent_t, _, latent_h, latent_w = latent.shape
        output_path = output_dir / f"zero_pose_latent_{latent_t}_{latent_h}_{latent_w}.pt"
        if output_path.exists() and not args.overwrite:
            logging.info("Skip existing %s", output_path)
            continue
        torch.save(latent, output_path)
        logging.info(
            "Saved %s from pixel shape [3, %d, %d, %d] -> latent shape %s",
            output_path,
            length,
            height,
            width,
            tuple(latent.shape),
        )

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
