![sneaky icon](docs/icon.png)

# sneaky™

## Visual Dataset Intelligence

Sneaky analyzes image datasets and turns them into a visual dataset
intelligence report.

It helps answer:

- What is in your dataset?
- How diverse is it?
- What doesn't fit?
- How much visual redundancy is there?
- How does it compare with another dataset?

See an example report: [docs/Sneaky_Sample_Report.pdf](docs/Sneaky_Sample_Report.pdf)

---

## Try it now

Two sample datasets with 700 AI-generated JPEG images are included in
this repository:

```
dataset_samples/
├── sample_01
└── sample_02
```

You can run Sneaky against them immediately after installation — no need
to prepare your own dataset first.

---

## Requirements

- Windows 10/11
- NVIDIA GPU
- Recent NVIDIA drivers
- Python 3.11

Sneaky runs locally. Your images are not uploaded anywhere.

---

## Installation

Clone the repository:

```
git clone https://github.com/thecayflow/sneaky.git
```

Open the `sneaky` folder and run:

```
install.bat
```

The installer creates the Python environment and installs the required
dependencies.

---

## Run Sneaky

Double-click:

```
run.bat
```

Sneaky will open in your browser.

---

## Your first analysis

1. Enter the path to a folder containing images (or point it at
   `dataset_samples/sample_01`).
2. Choose the number of semantic axes.
3. Click **Analyze**.
4. Explore the radar and projection views.
5. Click points or axes to inspect the underlying images.
6. Generate the PDF report.

---

## What Sneaky produces

### Semantic composition
A radar chart of the semantic themes Sneaky finds in your dataset —
detected automatically, or defined by you as custom text axes — showing
how the images are distributed across them.

### Diversity
A treemap of how evenly (or unevenly) your dataset is split across those
themes, so a handful of dominant categories are easy to spot at a glance.

### Outliers
Images that don't clearly match any detected theme, surfaced with actual
thumbnails — not just a percentage — so you can decide whether they
belong in the dataset at all.

### Visual redundancy
Near-duplicate images (camera bursts, repeated crops, minor edits)
detected independently of semantic content, with the option to browse
them grouped by similarity.

### Dataset comparison
Overlay two datasets on the same radar, and get a single distributional
distance score between them — is this new batch of images similar to
your existing one, or meaningfully different?

### PDF report
A designed, shareable summary of everything above, generated in one
click.

---

## Status

Sneaky is currently in beta.

Feedback is welcome.
