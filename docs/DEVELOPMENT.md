# Development notes

Technical documentation for working on Sneaky's codebase — manual
installation (without `install.bat`), project structure, and design
decisions worth understanding before making changes. See the root
`README.md` for the product-facing overview and the normal install path.

## Manual installation (without install.bat)

If you'd rather set up the environment yourself instead of running
`install.bat`:

```powershell
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> The first `streamlit run app.py` launch can take 1-2 minutes (it imports
> the full AI stack — torch, transformers, umap-learn...). That's normal,
> it doesn't mean it's frozen.

For the PDF's full typography (Barlow / Barlow Condensed), download those
font families from Google Fonts and place the `.ttf` files in
`src/report/fonts/` — if they're not there, the PDF still generates fine,
just with a fallback font (Helvetica).

## Project structure

See the folder tree under `src/` — each module has a single responsibility
(ingestion, embeddings, axes, scoring, viz, similarity, report,
persistence). See `BACKLOG.md` for the current status of phases and
pending improvements.

## Design decisions

### Hierarchical clustering: `average` → `ward` (with PCA)

**When**: while building `src/axes/hierarchical.py`.

**What happened**: the first version of `HierarchicalAxisEngine` used
`average` linkage over cosine distance, directly on the 512-dimensional
embedding space (CLIP `ViT-B-32`'s output). Tested on the real dataset
(`E:\dataset_unificado`, 3,092 images), the result was very unbalanced:

```
k=3: sizes=[3081, 6, 5]
k=5: sizes=[3069, 9, 6, 5, 3]
k=7: sizes=[3064, 9, 6, 5, 3, 3, 2]
```

One cluster swallowed almost the entire dataset, and the rest were tiny
clusters — not useful semantic blocks for the radar (selfies,
landscapes...), just atypical outlier images in the dataset.

**Why it happened**: `average` linkage over cosine distance in
high-dimensional spaces suffers from a phenomenon known as *chaining*:
instead of forming compact groups, it "chains" points one by one onto the
largest cluster.

**Fix applied**:
1. Change the default linkage method to **`ward`** (minimizes intra-cluster
   variance when merging — much more resistant to chaining).
2. Reduce embeddings from 512 to **50 dimensions via PCA** before
   clustering (`ward` needs to work in a real Euclidean space, and
   reducing dimensionality also helps the clustering come out better
   formed). Each cluster's centroid is still computed in the original
   space — PCA is only used to decide the grouping, not to represent the
   axes themselves.

**Result**: with `ward` + PCA, the same test gave (k=5):
`sizes=[1063, 669, 658, 356, 346]` — much more balanced and semantically
reasonable.

**Where it's configured** (`src/axes/hierarchical.py`):

```python
HierarchicalAxisEngine(
    linkage_method="ward",   # was: "average"
    pca_components=50,       # None to disable the PCA reduction
)
```

Both parameters are configurable when instantiating
`HierarchicalAxisEngine`, and also via a toggle in the UI itself.

### Hugging Face models and the `torch.load` block (torch < 2.6)

**When**: while building `src/axes/labeling.py`.

**What happened**: loading the BLIP captioning model
(`Salesforce/blip-image-captioning-base`) via `transformers` failed with:

```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with
`weights_only=True`, we now require users to upgrade torch to at least
v2.6 in order to use the function...
```

**Why it happens**: recent `transformers` versions block loading `.bin`
checkpoints (pickle-based, via `torch.load`) if the installed PyTorch is
below 2.6, due to a known security vulnerability (CVE-2025-32434). This
project uses PyTorch 2.5.1 (pinned by the CUDA/GPU version), so this block
applies.

**Fix applied**: force the **`.safetensors`** format instead of `.bin` — a
weights format that doesn't use pickle, so it isn't subject to that
restriction, and doesn't require touching the PyTorch version:

```python
BlipForConditionalGeneration.from_pretrained(model_name, use_safetensors=True)
```

**Why it matters going forward**: any new model loaded from Hugging Face in
this project can hit the same block if that model's repository doesn't
publish `.safetensors` weights by default. Adding `use_safetensors=True`
to the relevant `from_pretrained(...)` call is the standard fix.

### Incremental cache updates

**When**: when considering what happens if a new image is added to an
already-processed dataset.

**What used to happen**: `pipeline.py` only checked "does a cache exist for
this path?" and, if so, used it as-is — without comparing it against the
folder's actual contents. Adding or removing images had no effect until
the cache was deleted by hand and everything was recomputed from scratch
(~7 minutes).

**Fix applied**: `pipeline.py` now always scans the folder before deciding
anything, and compares the current image list against what was cached
(`src/pipeline.py::_update_embeddings_incrementally`):
- If nothing changed, the cache is used as-is (fast, as always).
- If there are new images, embeddings are generated *only* for those — the
  rest are reused without recomputation.
- If images were deleted, their embeddings are dropped from the cache.
- In either of the last two cases, the hierarchical tree and the per-k
  labels are invalidated (`cache.invalidate_tree_and_axes`), since they
  were computed from the old dataset composition — but this recomputes
  quickly (tens of seconds), since the real expense is the embeddings, not
  the clustering or labeling.

The cache can also be managed manually from the UI itself ("Cache
management", at the bottom of the page) without touching disk by hand.

### DataLoader with `num_workers>0` failing on Windows/Streamlit

**When**: after pushing the project to GitHub and testing it with a new
dataset.

**What happened**: while embedding images, the process failed with:

```
OSError: [WinError 6] The handle is invalid
  File "...multiprocessing\spawn.py", line 113, in spawn_main
    new_handle = reduction.duplicate(pipe_handle, ...)
```

**Why it happens**: `ClipEmbedder.embed_images` uses a PyTorch `DataLoader`
with several parallel processes (`num_workers=4` at the time) to
decode/preprocess images while the GPU works on the previous batch. On
Windows, this uses the `spawn` start method — each worker is a whole new
Python interpreter, not a lightweight `fork` like on Linux/macOS — and
under a process launched by Streamlit, the Windows handle duplication
needed to spin up those workers failed intermittently.

**Fix applied**: `DEFAULT_NUM_WORKERS` in `src/embeddings/clip_embedder.py`
went from `4` to `0` (no parallelism, single process) — slower at loading
images, but without this class of failure. If this is ever deployed on
Linux/macOS (or on a Windows setup where parallel loading is confirmed
stable), raising this value again is safe to try.

### `BlipForConditionalGeneration` failing with "Cannot copy out of meta tensor"

**When**: when re-analyzing the same dataset with a different `k`, within
the same app session.

**What happened**: loading BLIP (`ClusterLabeler.__init__`) failed with:

```
Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty()
instead of torch.nn.Module.to() when moving module from meta to a different device.
```

Not on every run — it worked the first time and failed on subsequent
re-analyses within the same process.

**Why it happens**: with `accelerate` installed, `transformers` loads
models via a lazier path by default: it first creates the model on a
special "meta" device (tensor shapes only, no real data, to save RAM
during loading) and fills it in afterward. BLIP's checkpoint is missing one
specific parameter (`text_decoder.cls.predictions.decoder.bias` — shown as
`MISSING` in the log, newly re-initialized) that's also **tied** to another
layer in the model. That specific parameter could end up left on the meta
device without being fully materialized, and `.to(device)` would then fail
since it had no real data to copy.

**Fix applied**, in `src/axes/labeling.py::ClusterLabeler.__init__`:
1. `low_cpu_mem_usage=False` in `from_pretrained(...)` — disables the
   meta-device loading path directly (on its own, this wasn't enough).
2. An explicit `self.model.tie_weights()`, right after `from_pretrained`
   and before `.to(self.device)` — forces the tied parameter to resolve
   correctly before moving the model, which is what actually fixed the
   failure.

**Why it matters going forward**: if another Hugging Face model with tied
weights gets added (common in text-generation heads that share memory with
the embedding layer), it's worth applying the same pattern
(`low_cpu_mem_usage=False` + `tie_weights()` before `.to(device)`) if this
same error shows up.

### A misplaced `@st.dialog` decorator masquerading as unrelated bugs

**When**: while building the "Copy images to..." feature (copies an
axis's images to a folder, for building sample datasets).

**What happened**: clicking "Copy images to..." opened an empty modal
titled "Axis images", showed the literal text "None" in a green box, and
the modal couldn't be closed. The images DID get copied correctly to disk
every time — only the UI feedback was broken.

**Why it happened**: `@st.dialog("Axis images", ...)` had ended up
decorating `_copy_axis_images_to` (the plain copy function) instead of
`show_axis_images_dialog` (the actual "View images" dialog) — a
refactoring slip from an earlier design iteration. `@st.dialog` turns a
function into a dialog trigger, not a normal callable: calling
`_copy_axis_images_to(...)` was silently interpreted as "open a dialog
named 'Axis images'", and its `return message` never reached the caller —
`message = _copy_axis_images_to(...)` evaluated to `None`, which is what
`st.success(None)` was rendering as the literal text "None".

**Why it matters going forward**: this produced misleading symptoms across
several rounds of debugging (looked like a session-state ordering issue, a
dialog self-close reliability issue, a tkinter/Streamlit conflict) because
every individual observation was consistent with those theories too. When
a plain function starts behaving like it opens a dialog (title matches, a
return value goes missing), check the decorators above it before assuming
the bug is in the function's own logic.
