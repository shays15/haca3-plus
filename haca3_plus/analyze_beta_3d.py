#!/usr/bin/env python3
"""Standalone HACA3+ 3D beta diagnostic. Run from the project parent directory:
python -m haca3_plus.analyze_beta_3d --checkpoint /path/model.pt --dataset-dirs /path/overfit_A /path/overfit_B --out-dir /path/results
Or place this script beside the haca3_plus package and run: python analyze_beta_3d.py ...
"""
import argparse
import csv
import itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description='Compare 3D beta logits, probabilities and labels across MRI contrasts')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset-dirs', nargs='+', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--mode', default='valid', choices=['train', 'valid'])
    p.add_argument('--contrasts', nargs='+', default=['T1PRE', 'T2', 'FLAIR', 'PD'])
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--beta-dim', type=int, default=5)
    p.add_argument('--base-ch', type=int, default=16)
    p.add_argument('--patch-size', type=int, default=128, help='Side length of center crop; use 0 for full volume')
    p.add_argument('--slice', type=int, default=-1, help='Axial/depth index within crop; -1 for center')
    p.add_argument('--gumbel-samples', type=int, default=20)
    p.add_argument('--normalization', default='01')
    return p.parse_args()


def load_classes():
    try:
        from haca3_plus.modules.network import UNet3d
        from haca3_plus.modules.dataset import HACA3Dataset
    except ImportError:
        try:
            from haca3_plus.network import UNet3d
            from haca3_plus.dataset import HACA3Dataset
        except ImportError as exc:
            raise ImportError('Run from the directory containing the haca3_plus package, or use python -m haca3_plus.analyze_beta_3d') from exc
    return UNet3d, HACA3Dataset


def metric_pair(a, b, mask):
    # a,b: [C,D,H,W] logits on CPU; mask [D,H,W]
    va = a[:, mask].float().T.contiguous()  # [V,C]
    vb = b[:, mask].float().T.contiguous()
    if va.shape[0] == 0:
        return None
    pa = torch.softmax(va, dim=1)
    pb = torch.softmax(vb, dim=1)
    la = pa.argmax(dim=1)
    lb = pb.argmax(dim=1)
    ca = va - va.mean(dim=1, keepdim=True)
    cb = vb - vb.mean(dim=1, keepdim=True)
    m = (pa + pb) * 0.5
    js = 0.5 * ((pa * (pa.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1) +
                (pb * (pb.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1)).mean()
    return dict(n_voxels=int(mask.sum()), label_agreement=float((la == lb).float().mean()),
                probability_mae=float((pa - pb).abs().mean()),
                centered_logit_mae=float((ca - cb).abs().mean()),
                raw_logit_mae=float((va - vb).abs().mean()),
                js_divergence=float(js),
                confidence_a=float(pa.max(dim=1).values.mean()),
                confidence_b=float(pb.max(dim=1).values.mean()))


def save_plots(probs, names, mask, subject, outdir, slice_idx):
    n = len(names)
    if n == 0:
        return
    k = probs[0].shape[0]
    fig, axes = plt.subplots(n, k, figsize=(3*k, 2.8*n), squeeze=False)
    for i, (p, name) in enumerate(zip(probs, names)):
        for c in range(k):
            im = axes[i,c].imshow(p[c, slice_idx].numpy(), cmap='viridis', vmin=0, vmax=1)
            axes[i,c].set_title('%s | P(beta=%d)' % (name,c), fontsize=9)
            axes[i,c].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.5, label='Probability')
    fig.savefig(str(outdir / ('subject_%03d_probabilities.png' % subject)), dpi=140, bbox_inches='tight')
    plt.close(fig)

    labels = [p.argmax(dim=0) for p in probs]
    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 4), squeeze=False)
    for i, (lab, name) in enumerate(zip(labels,names)):
        axes[0,i].imshow(lab[slice_idx].numpy(), cmap='tab10', vmin=0, vmax=k-1, interpolation='nearest')
        axes[0,i].set_title(name+' deterministic labels')
        axes[0,i].axis('off')
    fig.savefig(str(outdir / ('subject_%03d_labels.png' % subject)), dpi=140, bbox_inches='tight')
    plt.close(fig)

    pairs = list(itertools.combinations(range(n),2))
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(3.2*len(pairs), 3.6), squeeze=False)
        for i,(a,b) in enumerate(pairs):
            disagree = (labels[a][slice_idx] != labels[b][slice_idx]) & mask[slice_idx]
            axes[0,i].imshow(disagree.numpy(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[0,i].set_title('%s vs %s\nwhite = disagreement' % (names[a],names[b]), fontsize=9)
            axes[0,i].axis('off')
        fig.savefig(str(outdir / ('subject_%03d_disagreement.png' % subject)), dpi=140, bbox_inches='tight')
        plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    UNet3d, HACA3Dataset = load_classes()
    model = UNet3d(in_ch=1, out_ch=args.beta_dim, base_ch=args.base_ch, final_act='none').to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt['beta_encoder'] if 'beta_encoder' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print('Loaded beta encoder from:', args.checkpoint)
    print('Checkpoint epoch:', ckpt.get('epoch', 'unknown') if isinstance(ckpt,dict) else 'unknown')
    dataset = HACA3Dataset(dataset_dirs=args.dataset_dirs, contrasts=args.contrasts,
                           mode=args.mode, normalization_method=args.normalization,
                           patch_size=None if args.patch_size == 0 else (args.patch_size,)*3)
    rows = []
    occupancy = []
    gumbel_rows = []
    with torch.no_grad():
        for subj in range(len(dataset)):
            items = dataset[subj]
            valid_items = [(name, item) for name,item in zip(args.contrasts, items)
                           if bool(torch.as_tensor(item['exists']).item())]
            if len(valid_items) < 2:
                print('Skipping subject %d: fewer than 2 available contrasts' % subj)
                continue
            names, logits, masks, probs = [], [], [], []
            for name, item in valid_items:
                image = item['image'].unsqueeze(0).to(device)  # [1,1,D,H,W]
                logit = model(image)[0].float().cpu()         # [5,D,H,W]
                mask = item['mask'][0].bool().cpu()            # [D,H,W]
                p = torch.softmax(logit, dim=0)
                names.append(name)
                logits.append(logit)
                masks.append(mask)
                probs.append(p)
                labels = p.argmax(dim=0)
                counts = torch.bincount(labels[mask].reshape(-1), minlength=args.beta_dim).float()
                freq = counts / counts.sum().clamp_min(1)
                occupancy.append(dict(subject=subj, contrast=name, **{'beta_%d_fraction' % c: float(freq[c]) for c in range(args.beta_dim)}))
                # Sample Gumbel labels directly from the same fixed logits, foreground only.
                if args.gumbel_samples > 0:
                    agreements = []
                    for _ in range(args.gumbel_samples):
                        sampled = F.gumbel_softmax(logit, tau=1.0, hard=True, dim=0).argmax(dim=0)
                        agreements.append(float((sampled[mask] == labels[mask]).float().mean()))
                    gumbel_rows.append(dict(subject=subj, contrast=name,
                                            mean_gumbel_vs_argmax_agreement=float(np.mean(agreements)),
                                            std_gumbel_vs_argmax_agreement=float(np.std(agreements))))
                del image
            for a,b in itertools.combinations(range(len(names)),2):
                valid_mask = masks[a] & masks[b]
                result = metric_pair(logits[a], logits[b], valid_mask)
                if result:
                    rows.append(dict(subject=subj, contrast_a=names[a], contrast_b=names[b], **result))
                    print('Subject %d %-6s vs %-6s | agreement %.3f | prob MAE %.4f | centered logit MAE %.4f | JS %.4f' %
                          (subj, names[a], names[b], result['label_agreement'], result['probability_mae'],
                           result['centered_logit_mae'], result['js_divergence']))
            sl = args.slice if args.slice >= 0 else probs[0].shape[1]//2
            if not (0 <= sl < probs[0].shape[1]):
                raise ValueError('Slice %d outside volume depth %d' % (sl, probs[0].shape[1]))
            common_mask = torch.stack(masks).all(dim=0)
            save_plots(probs, names, common_mask, subj, outdir, sl)

    def write_csv(name, records):
        if records:
            with open(str(outdir/name), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
    write_csv('beta_pairwise_metrics.csv', rows)
    write_csv('beta_category_occupancy.csv', occupancy)
    write_csv('beta_gumbel_stability.csv', gumbel_rows)
    print('\nSaved results in:', outdir)

if __name__ == '__main__':
    main()
