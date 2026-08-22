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


def main() -> int:
    print("Checking sneaky installation...\n")

    # Critical: any of these failing means the app genuinely can't run —
    # these decide the exit code that install.bat checks.
    critical_results = [
        _check("PyTorch", "torch"),
        _check("OpenCLIP", "open_clip"),
        _check("Transformers", "transformers"),
        _check("spaCy", "spacy"),
        _check("Streamlit", "streamlit"),
        _check("UMAP", "umap"),
        _check("ReportLab", "reportlab"),
        _check("Matplotlib", "matplotlib"),
    ]

    # Informative, not critical: no CUDA-capable GPU means the app still
    # runs — just slowly (see README) — so this is shown to the user but
    # deliberately left OUT of the pass/fail exit code below. A missing
    # GPU shouldn't halt an otherwise-successful install.bat run. Printed
    # in the same position as before (right after the library checks,
    # before the spaCy model check), just tracked separately from here on.
    has_cuda = _check_cuda()

    critical_results.append(_check_spacy_model())

    print()
    if all(critical_results):
        print("sneaky installation successful!")
        if not has_cuda:
            print(
                "Note: no CUDA-capable GPU was detected — the app will still run, "
                "but noticeably slower (not tested on CPU-only setups)."
            )
        return 0

    print(
        "Some checks failed — see above. The app may still partly work, "
        "but re-run install.bat or check README.md if something looks wrong."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
