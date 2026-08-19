"""
check_install.py

Standalone installation sanity check for sneaky(TM) Semantic Report — run
automatically at the end of install.bat, or any time by hand with:

    python check_install.py

Tries importing each library the app depends on, one at a time, and
prints a simple OK/FAIL checklist. The CUDA/GPU line is a REAL check (not
cosmetic) — it actually asks PyTorch whether a usable NVIDIA GPU is
present, since that's the single most common thing to go wrong.
"""

import importlib

LABEL_WIDTH = 28


def _check(label: str, import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        print(f"  {label:.<{LABEL_WIDTH}} OK")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:.<{LABEL_WIDTH}} FAIL  ({exc})")
        return False


def _check_cuda() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  {'CUDA / NVIDIA GPU':.<{LABEL_WIDTH}} OK    ({gpu_name})")
            return True
        print(
            f"  {'CUDA / NVIDIA GPU':.<{LABEL_WIDTH}} FAIL  "
            "(no CUDA-capable GPU detected — the app will run, but very slowly)"
        )
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  {'CUDA / NVIDIA GPU':.<{LABEL_WIDTH}} FAIL  ({exc})")
        return False


def _check_spacy_model() -> bool:
    # spaCy's language model is a SEPARATE download from the spacy package
    # itself (`python -m spacy download en_core_web_sm`) — check it can
    # actually be loaded, not just that the spacy package exists.
    try:
        import spacy

        spacy.load("en_core_web_sm")
        print(f"  {'spaCy language model':.<{LABEL_WIDTH}} OK")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  {'spaCy language model':.<{LABEL_WIDTH}} FAIL  ({exc})")
        return False


def main() -> None:
    print("Checking sneaky installation...\n")

    results = [
        _check("PyTorch", "torch"),
        _check("OpenCLIP", "open_clip"),
        _check("Transformers", "transformers"),
        _check("spaCy", "spacy"),
        _check("Streamlit", "streamlit"),
        _check("UMAP", "umap"),
        _check("ReportLab", "reportlab"),
        _check("Matplotlib", "matplotlib"),
        _check_cuda(),
        _check_spacy_model(),
    ]

    print()
    if all(results):
        print("sneaky installation successful!")
    else:
        print(
            "Some checks failed — see above. The app may still partly work, "
            "but re-run install.bat or check README.md if something looks wrong."
        )


if __name__ == "__main__":
    main()
