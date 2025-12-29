import sys
import subprocess


def test_dl_batch_requires_torch_message():
    # Simulate missing torch by injecting a None module in a child process
    code = r"""
import sys, types
from pathlib import Path
sys.modules['torch'] = None
from hfoGUI.cli import run_dl_batch
args = types.SimpleNamespace(
  file=str(Path('.')),
  set_file=None,
  output=None,
  model_path='model.pt',
  threshold=0.5,
  batch_size=32,
  skip_bits2uv=True,
  verbose=False,
)
run_dl_batch(args)
"""
    proc = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = proc.stdout.decode(errors="ignore") + proc.stderr.decode(errors="ignore")
    assert "Deep Learning detection requires PyTorch" in out, f"Expected guidance message not shown. Output:\n{out}"
    assert proc.returncode == 0, f"Child process failed with code {proc.returncode}. Output:\n{out}"
