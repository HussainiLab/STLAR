import argparse
import torch
import numpy as np
from pathlib import Path
from .model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Export HFO model to TorchScript and ONNX")
    p.add_argument('--ckpt', required=True, help='Path to best.pt checkpoint')
    p.add_argument('--onnx', required=True, help='Output ONNX path')
    p.add_argument('--ts', required=True, help='Output TorchScript path (.pt)')
    p.add_argument('--example-len', type=int, default=2000, help='Example segment length for tracing')
    p.add_argument('--model-type', type=int, default=2, help='Model architecture: 1=SimpleCNN, 2=ResNet1D, 3=InceptionTime, 4=Transformer, 5=2D_CNN')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cpu')

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    
    # Try to infer model type from checkpoint if available, else use arg
    model_type = ckpt.get('model_type', args.model_type)
    
    model = build_model(model_type)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    ex_len = int(args.example_len)
    dummy = torch.randn(1, 1, ex_len, device=device)

    # TorchScript
    ts = torch.jit.trace(model, dummy)
    ts.save(args.ts)
    print(f"✓ Saved TorchScript to {args.ts}")

    # ONNX (optional - skip if onnx not installed)
    try:
        onnx_path = Path(args.onnx)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model, dummy, args.onnx,
            input_names=['input'], output_names=['logits'],
            opset_version=14,
            dynamic_axes={'input': {0: 'batch', 2: 'length'}, 'logits': {0: 'batch'}},
        )
        print(f"✓ Saved ONNX to {args.onnx}")
    except ImportError:
        print("⚠️  ONNX module not installed; skipping ONNX export. Install with: pip install onnx")
    except Exception as e:
        print(f"⚠️  ONNX export failed ({e}); TorchScript is available and ready for inference.")


if __name__ == '__main__':
    main()
