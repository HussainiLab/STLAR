import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data import SegmentDataset, pad_collate_fn
from .model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Train a 1D CNN HFO classifier")
    p.add_argument('--train', required=True, help='Path to train manifest CSV')
    p.add_argument('--val', required=True, help='Path to val manifest CSV')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--out-dir', type=str, default='models')
    p.add_argument('--num-workers', type=int, default=2)
    return p.parse_args()


def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logit = model(x)
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / max(total, 1)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
    return total_loss / max(total, 1)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = SegmentDataset(args.train)
    val_ds = SegmentDataset(args.val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=pad_collate_fn)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('inf')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / 'best.pt'

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict()}, best_ckpt)
            print(f"Saved best checkpoint to {best_ckpt}")

    print("Done. Best val loss: {:.4f}".format(best_val))


if __name__ == '__main__':
    main()
