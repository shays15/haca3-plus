"""Full-volume, no-stitching inference for the simplified 3D HACA3+ model.

Place this file in haca3_plus/test_full_volume.py and run with
python -m haca3_plus.test_full_volume ... from the package's parent directory.
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .modules.network import UNet3d, ThetaEncoder3d


def load_volume(path, normalization='01'):
    nii = nib.load(str(path))
    vol = nii.get_fdata(dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f'Expected 3D NIfTI, got {vol.shape}: {path}')
    if normalization == '01':
        scale = float(np.percentile(vol, 95))
        vol = np.clip(vol / max(scale, 1e-5), 0, 5)
    elif normalization == 'wm':
        scale = 2.0
        vol = vol / scale
    else:
        scale = 1.0
    tensor = torch.from_numpy(np.ascontiguousarray(vol))[None, None]
    return tensor, nii, scale


def save_nifti(data, ref, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), ref.affine,
                             header=header), str(path))
    print(f'Saved {path}: {data.shape}')


def load_weights(module, state, name):
    if name not in state:
        raise KeyError(f'Checkpoint missing {name!r}; available keys: {list(state)}')
    module.load_state_dict(state[name], strict=True)
    module.eval()


def main():
    p = argparse.ArgumentParser(description='Full-volume 3D HACA3+ reconstruction, no stitching')
    p.add_argument('--in-path', type=Path, required=True, help='Source MRI (full 3D NIfTI)')
    p.add_argument('--target-image', type=Path, help='MRI to encode target contrast; omit to use source')
    p.add_argument('--checkpoint', type=Path, required=True, help='HACA3 training checkpoint')
    p.add_argument('--theta-checkpoint', type=Path, help='Optional separately pretrained theta checkpoint')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--beta-dim', type=int, default=5)
    p.add_argument('--theta-dim', type=int, default=2)
    p.add_argument('--normalization-method', choices=['01', 'wm', 'none'], default='01')
    p.add_argument('--beta-mode', choices=['softmax', 'hard'], default='softmax',
                   help='Deterministic probabilities or one-hot argmax beta')
    p.add_argument('--fp16', action='store_true', help='Use mixed precision on CUDA to reduce VRAM')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source, ref, source_scale = load_volume(args.in_path, args.normalization_method)
    target, _, _ = load_volume(args.target_image or args.in_path, args.normalization_method)
    if source.shape[2:] != target.shape[2:]:
        print('Note: target and source differ in shape; only target theta is used.')
    print('Source shape:', tuple(source.shape), 'device:', device)
    print('Target shape:', tuple(target.shape))

    beta_encoder = UNet3d(in_ch=1, out_ch=args.beta_dim, base_ch=16, final_act='none').to(device)
    theta_encoder = ThetaEncoder3d(in_ch=1, out_ch=args.theta_dim).to(device)
    decoder = UNet3d(in_ch=1 + args.theta_dim, out_ch=1, base_ch=16, final_act='relu').to(device)

    # Supports checkpoints saved as {'beta_encoder': ..., 'decoder': ...}.
    # Loading explicitly is important: the current HACA3 constructor does not
    # implement pretrained_haca3 checkpoint loading.
    state = torch.load(args.checkpoint, map_location='cpu')
    load_weights(beta_encoder, state, 'beta_encoder')
    load_weights(decoder, state, 'decoder')
    theta_state = (torch.load(args.theta_checkpoint, map_location='cpu')
                   if args.theta_checkpoint else state)
    load_weights(theta_encoder, theta_state, 'theta_encoder')
    del state, theta_state

    source = source.to(device)
    target = target.to(device)
    try:
        with torch.inference_mode():
            # Encoder forward passes are separated to minimize peak GPU memory.
            with torch.autocast(device_type='cuda', enabled=args.fp16 and device.type == 'cuda'):
                theta_mu, _ = theta_encoder(target)
            print('Theta mean:', theta_mu.float().cpu().numpy().tolist())
            del target

            with torch.autocast(device_type='cuda', enabled=args.fp16 and device.type == 'cuda'):
                logits = beta_encoder(source)
            del source

            probabilities = F.softmax(logits.float(), dim=1)
            if args.beta_mode == 'hard':
                labels = probabilities.argmax(dim=1, keepdim=True)
                beta = labels.float() / args.beta_dim
            else:
                channel_values = torch.arange(args.beta_dim, device=device, dtype=torch.float32)
                beta = (probabilities * channel_values[None, :, None, None, None]).sum(dim=1, keepdim=True) / args.beta_dim

            # Save the anatomical representation BEFORE decoding.
            save_nifti(beta[0, 0].cpu().numpy(), ref, args.out_dir / 'beta_aggregated.nii.gz')
            save_nifti(probabilities[0].permute(1, 2, 3, 0).cpu().numpy(),
                       ref, args.out_dir / 'beta_probabilities.nii.gz')
            del probabilities, logits

            D, H, W = beta.shape[2:]
            theta_map = theta_mu.float()[:, :, None, None, None].expand(-1, -1, D, H, W)
            decoder_input = torch.cat([beta, theta_map], dim=1)
            del beta, theta_map
            with torch.autocast(device_type='cuda', enabled=args.fp16 and device.type == 'cuda'):
                reconstruction = decoder(decoder_input)
            del decoder_input
            output = reconstruction[0, 0].float().cpu().numpy() * source_scale
            save_nifti(output, ref, args.out_dir / 'reconstruction.nii.gz')
            print('Reconstruction range:', float(output.min()), float(output.max()))
    except torch.cuda.OutOfMemoryError:
        print('CUDA OOM on full volume. Retry with --fp16 or use a larger GPU.')
        raise


if __name__ == '__main__':
    main()
