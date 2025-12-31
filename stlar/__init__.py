import importlib
import sys
import re
import pathlib

# Lightweight version extraction (no heavy imports)
try:
    _root = pathlib.Path(__file__).parent.parent
    _main_text = (_root / 'hfoGUI' / '__main__.py').read_text(encoding='utf-8')
    _m = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", _main_text)
    version = _m.group(1) if _m else '0.0.0'
except Exception:
    version = '0.0.0'

# Alias key submodules so imports like `stlar.core` resolve
# NOTE: We avoid importing 'exporters' at module level because it depends on Qt/pyqtgraph
# and requires QApplication to be created first. Instead, use lazy imports.
for sub in [
    'core',
    'dl_training',
    'intan_rhd_format',
]:
    sys.modules[f'stlar.{sub}'] = importlib.import_module(f'hfoGUI.{sub}')

# Convenience: expose `run` like original (may require GUI deps at runtime)
try:
    from hfoGUI.main import run  # noqa: F401
except Exception:
    pass

