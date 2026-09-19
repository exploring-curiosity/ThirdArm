#!/usr/bin/env python3
"""Re-apply the Apple Silicon patches SAM 3 needs to run on MPS.

`pip install sam3` ships code that only runs on CUDA machines. Three things
break on Apple Silicon, and all three are fixed here. The patches live in
site-packages, so **any reinstall or venv rebuild silently reverts them** and
auto_pose.py starts failing in confusing ways. Run this after any such change:

    python patch_sam3.py

It is idempotent: re-running on an already-patched install reports "ok" and
changes nothing. With --check it only reports, exiting 1 if a patch is missing,
which makes it usable as a preflight step.

The three problems, in the order you hit them:

1. `sam3/model/edt.py` imports triton at module scope. triton ships CUDA
   kernels and has no Apple Silicon wheel, so the import aborts before any
   model loads. The function it decorates, edt_triton, is only called during
   training-time click simulation, never at inference, so a scipy-based CPU
   fallback is enough.

2. `sam3/model/position_encoding.py` and `sam3/model/decoder.py` allocate
   helper tensors with a hardcoded `device="cuda"`. On this machine that
   raises immediately. Replaced with a helper preferring MPS, then CUDA, then
   CPU, so the same source runs on either machine.

3. The CLIP BPE vocabulary asset is missing from the wheel. The tokenizer
   looks for `assets/bpe_simple_vocab_16e6.txt.gz` next to the package root
   and raises FileNotFoundError. This script cannot fetch it offline; if it is
   absent, the instructions to restore it are printed.

Two further traps are NOT patched here, because they are in our calling code
rather than the library. They are recorded in docs/sam3-install-notes.md:
  - build_sam3_image_model(device=...) accepts the argument and ignores it;
    you must call .to(device) on the returned model yourself.
  - set_image() requires a PIL Image. Handed an HWC numpy array it does not
    raise; it silently returns empty masks with a nonsense shape like
    (2, 1, 960, 3), which looks like a segmentation failure, not a type error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEVICE_HELPER = '''def _default_device():
    """Device to allocate helper tensors on.

    SAM 3 hardcodes "cuda" here, which aborts on Apple Silicon. Prefer MPS,
    then CUDA, then CPU, so the same code runs on either machine.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"




'''

TRITON_GUARD = '''
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CUDA-only dependency
    # triton ships CUDA kernels and does not install on Apple Silicon.
    # edt_triton is only reached from training-time click simulation
    # (sam3_tracker_utils compares predictions against ground-truth masks),
    # so inference never calls it. A scipy fallback is defined at the bottom
    # of this module; the stubs below keep the @triton.jit decoration valid.
    _HAS_TRITON = False

    class _TL:
        constexpr = int

    tl = _TL()

    class _Triton:
        @staticmethod
        def jit(fn=None, **kwargs):
            return fn if fn is not None else (lambda f: f)

    triton = _Triton()
'''

SCIPY_FALLBACK = '''


if not _HAS_TRITON:  # pragma: no cover - non-CUDA fallback
    def edt_triton(data: torch.Tensor):  # noqa: F811
        """Exact Euclidean distance transform on CPU, via scipy.

        Same semantics as the triton kernel (distance to the nearest zero,
        computed per batch item). This path is training-only, so inference
        speed is unaffected.
        """
        from scipy import ndimage

        arr = data.detach().to("cpu").numpy()
        out = np.stack([ndimage.distance_transform_edt(a) for a in arr])
        return torch.from_numpy(out).to(device=data.device, dtype=torch.float32)
'''

BPE_NAME = "bpe_simple_vocab_16e6.txt.gz"
BPE_URL = (
    "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
)
BPE_EXPECTED_BYTES = 1_356_917


def sam3_root() -> Path:
    """Locate the installed sam3 package without importing it.

    Importing would run the very code that is broken before patching.
    """
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / "sam3"
        if (candidate / "model").is_dir():
            return candidate
    raise SystemExit(
        "sam3 is not installed in this interpreter. Activate the venv first:\n"
        "    source .venv/bin/activate && python patch_sam3.py"
    )


def patch_device(path: Path, check: bool) -> bool:
    """Replace device="cuda" literals with a device-aware helper."""
    src = path.read_text()
    if "_default_device" in src:
        return True
    if check:
        return False

    if 'device="cuda"' not in src:
        raise SystemExit(
            f"{path.name}: expected a device=\"cuda\" literal but found none. "
            "The sam3 version has changed; re-derive the patch by hand."
        )

    # Insert the helper after the import block: before the first line that
    # starts a class or def at column 0.
    lines = src.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(("class ", "def ")):
            break
    else:
        raise SystemExit(f"{path.name}: no top-level class or def found.")

    patched = "".join(lines[:i]) + DEVICE_HELPER + "".join(lines[i:])
    patched = patched.replace('device="cuda"', "device=_default_device()")
    path.write_text(patched)
    return True


def patch_edt(path: Path, check: bool) -> bool:
    """Guard the triton import and add the scipy distance-transform fallback."""
    src = path.read_text()
    if "_HAS_TRITON" in src:
        return True
    if check:
        return False

    old = "import triton\nimport triton.language as tl\n"
    if old not in src:
        raise SystemExit(
            "edt.py: expected a bare triton import but found none. "
            "The sam3 version has changed; re-derive the patch by hand."
        )
    patched = src.replace(old, TRITON_GUARD, 1)

    # The scipy fallback below calls np.stack, but upstream edt.py does not
    # import numpy. Without this the fallback raises NameError the first time
    # it is reached, which would look like a scipy problem, not a missing
    # import.
    if "import numpy as np" not in patched:
        patched = patched.replace("import torch", "import numpy as np\nimport torch", 1)

    path.write_text(patched.rstrip("\n") + SCIPY_FALLBACK)
    return True


def check_bpe(root: Path) -> bool:
    """The CLIP vocabulary the tokenizer needs, missing from the wheel."""
    asset = root.parent / "assets" / BPE_NAME
    return asset.is_file() and asset.stat().st_size == BPE_EXPECTED_BYTES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="report only; exit 1 if any patch is missing",
    )
    args = ap.parse_args()

    root = sam3_root()
    print(f"sam3 at {root}")

    results = [
        ("edt.py triton guard", patch_edt(root / "model" / "edt.py", args.check)),
        (
            "position_encoding.py device",
            patch_device(root / "model" / "position_encoding.py", args.check),
        ),
        ("decoder.py device", patch_device(root / "model" / "decoder.py", args.check)),
        ("CLIP bpe asset", check_bpe(root)),
    ]

    for name, ok in results:
        print(f"  {'ok  ' if ok else 'MISS'}  {name}")

    if not check_bpe(root):
        print(
            f"\nThe tokenizer asset is missing. Restore it with:\n"
            f"    mkdir -p {root.parent / 'assets'}\n"
            f"    curl -L -o {root.parent / 'assets' / BPE_NAME} {BPE_URL}\n"
            f"Expected size: {BPE_EXPECTED_BYTES} bytes."
        )

    missing = [name for name, ok in results if not ok]
    if missing:
        if args.check:
            print(f"\n{len(missing)} patch(es) missing. Run without --check.")
        return 1

    print("\nAll patches present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
