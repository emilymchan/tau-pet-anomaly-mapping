from pathlib import Path
from importlib.metadata import distribution

timm_dir = Path(distribution("timm").locate_file("timm"))
helpers = timm_dir / "models" / "layers" / "helpers.py"

source = helpers.read_text()

old = "from torch._six import container_abcs"
new = "import collections.abc as container_abcs"

if old in source:
    helpers.write_text(source.replace(old, new))
    print("Patched timm for PyTorch 2.x.")
elif new in source:
    print("timm is already compatible.")
else:
    raise RuntimeError(f"Unexpected timm installation: {helpers}")