# Development notes

Technical documentation for working on sneakyReport's codebase — manual
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
persistence).

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
- If an EXISTING image's content changes in place (same filename, edited
  file — a modification time comparison, not just the path set, is what
  catches this), it's treated the same as a new image: only that one gets
  re-embedded, not the whole dataset. Each dataset's `mtimes.json` records
  what the file's modification time was the last time it was embedded — a
  dataset cached before this existed simply has no recorded mtime yet, and
  a missing entry is treated as "unchanged" rather than forcing a
  needless full re-embed the first time this runs on an older cache.
- In any of the above cases beyond "nothing changed", the hierarchical
  tree and the per-k labels are invalidated (`cache.invalidate_tree_and_axes`),
  since they were computed from the old dataset composition — but this
  recomputes quickly (tens of seconds), since the real expense is the
  embeddings, not the clustering or labeling.

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

### Scoring standardization drifting per-dataset when a comparison feed is active

**When**: while building out dataset comparison — an axis with genuinely
zero matching images in one dataset was still claiming images there.

**What happened**: with a comparison feed loaded, `scoring.py`'s dominant-
axis assignment (`get_dominant_labels` and everything built on it — axis
counts, the radar, "View images", the PDF) standardized (z-scored) each
dataset's raw cosine similarities against **its own** distribution before
comparing axes. Two concrete symptoms this caused:
- An axis with a real concept absent from one dataset (e.g. "owl", with
  zero actual owls in that dataset) could still get 20+ images assigned
  to it there — z-scoring within that dataset alone just picks out
  whichever axis happens to score *relatively* highest, even if every raw
  similarity is low across the board.
- A custom TEXT axis (e.g. "Horse", added by the user, scored against
  CLIP's text embedding rather than an image centroid) could win purely on
  z-score even when its raw cosine similarity to every image was
  genuinely too low to mean anything — text-vs-image CLIP similarities
  sit on a different absolute scale than image-vs-image ones (the
  "modality gap"), so a text axis's z-score isn't directly comparable to
  an auto-detected axis's.

**Why it happened**: standardizing each dataset against its own
distribution assumes the two distributions are equivalent enough to
compare directly — reasonable for a single dataset, not once a second
one with a genuinely different composition enters the picture. Each
dataset ends up graded on its own curve, so "best-scoring axis in THIS
dataset" and "best-scoring axis considering BOTH" can disagree.

**Fix applied**, in `scoring.py` and `app.py`:
1. `get_dominant_labels`, `get_axis_counts_by_dominance`,
   `get_ranked_images_for_axis`, and `get_radar_values_by_dominance` all
   accept an optional `standardize_reference` — a pooled score matrix to
   standardize against, instead of each dataset's own.
2. When a comparison feed is active, `app.py::_pooled_score_reference`
   computes this pooled reference **once**, from both datasets' combined
   scores, and passes it to every one of the ~9 call sites that score
   either dataset — found by grepping for all four function names above,
   not just the obvious ones (two PDF-specific call sites were missed on
   the first pass and only caught in a later round).
3. `CUSTOM_AXIS_MIN_SIMILARITY = 0.20` — a custom (text) axis must clear
   this **raw cosine floor** to claim an image at all; failing that, it's
   disqualified as a candidate for that image (which moves to its next-
   best axis, not straight to "Other") regardless of how favorable its
   z-score looks. This is what actually fixed the "Horse" case above.

**Why it matters going forward**: any NEW function added to `scoring.py`
that standardizes per-dataset similarities needs the same
`standardize_reference` treatment the moment dataset comparison is in
play — the bug doesn't announce itself with an exception, just with axes
quietly claiming images they have no real business claiming. When
debugging "why does this axis have images that don't belong," check
whether the call site in question received the pooled reference before
looking anywhere else.

### Combined axis extraction — re-clustering both datasets together

**When**: after Fase 2 (comparison scoring) was already working — the
user wanted a theme present in only ONE of two compared datasets (e.g.
many horses in the second feed, none at all in the first) to be able to
surface as its own axis, not just get silently absorbed into whichever
existing axis scored closest.

**What happened / why the change was needed**: axes were always computed
(hierarchical clustering + BLIP captioning) from the PRIMARY dataset
alone, before a comparison feed even entered the picture — a comparison
feed's images were only ever *scored* against those already-fixed axes
(`get_embeddings_only`, no clustering of its own). A concept entirely
absent from the primary dataset had no way to become its own axis, no
matter how much of the comparison dataset it made up.

**Design options considered**: (A) always re-cluster over BOTH datasets'
pooled embeddings whenever a comparison feed is active; (B) leave the
primary's own axes untouched, and only look for coherent "gap" groups
among the comparison feed's own leftover ("Other") images, suggesting new
axes asymmetrically; (C) an explicit "analyze as a comparison" mode where
both paths are required up front, before any clustering happens at all.
**Chosen: (A)** — the user judged it "the most honest" approach worth the
real cost tradeoff it implies (below), rather than a cheaper approximation.

**Fix applied**:
- `cache.py`: a cache location keyed by the PAIR of dataset paths, not
  just one (`_pair_cache_dir` — hashes the two paths **sorted**, so
  comparing A-vs-B and B-vs-A hit the exact same cache entry). Stores its
  own tree/axes (`save_pair_tree`/`load_pair_tree`/`save_pair_axes`/
  `load_pair_axes`), invalidated (`invalidate_pair_tree_and_axes`) when
  the pooled path set changes on either side.
- `pipeline.py::run_combined_pipeline`: fetches each dataset's OWN
  embeddings independently (reusing each one's existing per-dataset cache
  — analyzing A alone, or A combined with C, never repeats A's embedding
  pass), concatenates them, and runs the SAME clustering/captioning
  pipeline as a single-dataset analysis over the pooled result — the
  clustering itself needed zero changes to work on pooled embeddings from
  two sources instead of one.
- `app.py`: `full_axes`'s auto-detected component switches between the
  combined result (`combined_base_axes`, when a comparison is active and
  already combined at the current k/method) and the primary's own
  solo-dataset axes, falling back to solo automatically the moment the
  comparison is removed. Every downstream consumer (axis counts, radar,
  scatter, PDF, "View images") needed NO changes — they already operated
  generically on "whatever full_axes currently is."

**Real cost, by design**: unlike scoring a comparison feed against
already-fixed axes (cheap), this is comparable in cost to a first-time
Analyze the first time a given PAIR of datasets is combined at a given
k/linkage_method — full re-clustering + re-captioning, not just an
embeddings pass. Cached afterward per (pair, k, method), same as the
single-dataset pipeline. The user explicitly accepted this tradeoff
before implementation began.

**UI iteration worth knowing about**: the first version placed a "Load
comparison feed" button ABOVE "Analyze," meant only for adding a
comparison to an already-analyzed primary — but nothing stopped someone
who'd just filled in both paths from clicking it first, hitting a
confusing "analyze the primary first" error despite having just done
exactly that (in their own mental model). Fixed by only rendering that
button once a primary result already exists, then simplified further:
both path fields are always visible up front, "Analyze" alone handles
every case (solo, solo-then-add-later, combined-from-the-start) by
checking whether the second field has content, and a "Remove comparison
feed" button stays disabled until there's an actual comparison loaded to
remove.

**Known limitation**: pair-level cache invalidation currently checks the
combined PATH SET only. An image edited in place (same path, changed
content — caught by each dataset's own `mtimes.json`, see the
"Incremental cache updates" section above) does NOT currently invalidate
an already-cached combined tree/axes for that pair, since the path set
itself hasn't changed. Noted rather than silently left unhandled; low
priority given how rarely someone edits an image in place without
renaming it.

### Console noise from `transformers` about models this project never uses

**When**: while polishing the terminal output ahead of the public beta —
a dozen or so `[ERROR]` lines about `DeepseekVLHybridImageProcessorKwargs`,
`Kimi_K25ImageProcessorKwargs`, and `PaddleOCRVLImageProcessorKwargs`
appearing on every "Analyze," for model families this project has
never imported or used anywhere.

**What happened / dead ends worth knowing about**: several reasonable-
looking fixes were tried, in order, before finding the real cause — worth
recording so nobody repeats the same path:
1. `logging.getLogger("transformers").setLevel(...)`, first at `ERROR`
   then at `CRITICAL` — no effect. Turned out these lines are printed via
   a plain `print()`, not Python's `logging` module at all, so no logger
   configuration could ever touch them.
2. `contextlib.redirect_stdout`/`redirect_stderr` scoped tightly around
   every actual place `labeling.py` touches `transformers` (the import,
   both `from_pretrained()` calls, and the `model.generate()` inference
   call) — still no effect, which briefly (and incorrectly) suggested the
   print was happening at the OS file-descriptor level, bypassing Python
   entirely (e.g. from a Rust component).

**The real cause**, found by temporarily monkey-patching `sys.stdout`/
`sys.stderr` to dump a full call stack (`traceback.print_stack()`) the
moment one of these lines was written, rather than continuing to guess:
it has nothing to do with anything in this project's own code path.
Streamlit's own file watcher (`streamlit/watcher/local_sources_watcher.py`)
calls `hasattr(module, "__path__")` on every already-imported module on
**every script rerun**, to decide what to watch for auto-reload. For the
`transformers` module specifically — which defines its own lazy-import
`__getattr__` in `transformers/__init__.py` — that innocuous `hasattr()`
call triggers a full import of every registered model's image-processor
class, including ones this project never uses, each carrying transformers'
own `@auto_docstring` decorator, which prints this "undocumented
parameter" notice as a side effect of being imported at all.

**Fix applied**: `.streamlit/config.toml`, setting `fileWatcherType =
"none"` — disables Streamlit's file watcher outright, removing the
trigger entirely rather than filtering its symptom. Confirmed via an
isolation test (running with ONLY this config change, no code-level
filtering) that this alone is sufficient — no stdout/stderr filtering
code was needed in the end.

**Real tradeoff, not just a cosmetic setting**: this disables Streamlit's
auto-reload-on-file-save entirely. Editing any `.py` file while `run.bat`
is running no longer refreshes the browser automatically — restart it by
hand to pick up changes. This has zero effect on anyone just USING the
app (they never edit its source), but affects active development
directly: expect to manually restart after every code change from now
on. Normal widget-driven interactivity (clicking a button, a script
rerunning in response) is unrelated to this setting and unaffected — that
happens over the websocket connection, not the filesystem watcher.

**Why it matters going forward**: if new console noise shows up that
resists both logging configuration AND scoped stdout/stderr redirection
around wherever it seems to originate, consider whether it might not be
triggered by this project's own code at all — Streamlit's own file
watcher re-touching every loaded module on every rerun is a real, easy-
to-miss trigger for lazy-import side effects in large libraries.
