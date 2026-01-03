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
    p.add_argument('--use-cwt', action='store_true', help='Use CWT/Scalogram preprocessing for 2D models')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cpu')

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    
    # Try to infer model type and num_classes from checkpoint if available, else use arg
    model_type = ckpt.get('model_type', args.model_type)
    num_classes = ckpt.get('num_classes', 1)  # Default to 1 if not in checkpoint
    
    model = build_model(model_type, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    ex_len = int(args.example_len)
    
    # Create appropriate dummy input shape
    # 2D models (when use-cwt is enabled) expect (batch, channels, height, width)
    # 1D models expect (batch, channels, length)
    
    # Model type 6 (HFO_2D_CNN) expects 4D input (B, 1, F, T)
    is_2d_input = args.use_cwt or (model_type == 6)

    if is_2d_input:
        # 2D input: (batch=1, channels=1, freq_bins=64, time=example_len)
        dummy = torch.randn(1, 1, 64, ex_len, device=device)
    else:
        # 1D input: (batch=1, channels=1, length=example_len)
        dummy = torch.randn(1, 1, ex_len, device=device)

    # TorchScript
    ts = torch.jit.trace(model, dummy)
    ts.save(args.ts)
    print(f"✓ Saved TorchScript to {args.ts}")

    # ONNX (optional - skip if onnx not installed)
    try:
        onnx_path = Path(args.onnx)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Adjust dynamic axes based on input dimensionality
        if is_2d_input:
            # 2D input: dynamic batch and time dimensions
            dynamic_axes = {
                'input': {0: 'batch', 3: 'time'}, 
                'logits': {0: 'batch'}
            }
        else:
            # 1D input: dynamic batch and length dimensions
            dynamic_axes = {
                'input': {0: 'batch', 2: 'length'}, 
                'logits': {0: 'batch'}
            }
        
        torch.onnx.export(
            model, dummy, args.onnx,
            input_names=['input'], output_names=['logits'],
            opset_version=17,
            dynamic_axes=dynamic_axes,
        )
        print(f"✓ Saved ONNX to {args.onnx}")
    except ImportError:
        print("⚠️  ONNX module not installed; skipping ONNX export. Install with: pip install onnx")
    except RuntimeError as e:
        if "adaptive_avg_pool" in str(e).lower():
            print("⚠️  ONNX export skipped: adaptive pooling with dynamic input size not supported in ONNX")
            print("    (This is a PyTorch ONNX exporter limitation, not a bug in the model)")
            print("    → TorchScript is available and recommended for production inference")
        else:
            print(f"⚠️  ONNX export failed: {str(e)[:200]}")
            print("    → TorchScript is available and ready for inference")
    except Exception as e:
        print(f"⚠️  ONNX export failed ({type(e).__name__}); TorchScript is available and ready for inference.")


if __name__ == '__main__':
    main()
