Machine Learning Algorithms Implementations

A collection of hands‑on Jupyter notebooks that re‑implement core machine‑learning algorithms from scratch and walk through real‑world datasets, complete with exploratory analysis, model training, evaluation and speed/accuracy trade‑off experiments.

📂 Repository Contents

Folder / Notebook

Algorithm / Topic

Highlights

knn.ipynb

Brute‑force k‑Nearest Neighbours (PyTorch)

cosine / Euclidean distance kernels, batching on GPU

knn_ivf.ipynb

Inverted File (IVF) Approximate‑KNN

coarse quantisation, probing lists,  10× speed‑up benchmarks

knn_lsh.ipynb

Locality‑Sensitive Hashing (LSH)

random hyper‑plane hashes, Hamming buckets, recall vs probes curves

decision_tree.ipynb

Decision‑Tree Classifier

entropy vs Gini, categorical encoding, fraud‑detection case study

linear_regression.ipynb

Linear Regression

courier‑delivery dataset, feature scaling, residual diagnostics

Why these notebooks?  They illustrate the spectrum from exact to approximate neighbour search, and from classic interpretable models to modern scalable DL‑assisted pipelines.

Quick Start

Clone & install deps

git clone <repo‑url>
cd ml‑algorithms‑sandbox
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

Launch Jupyter

jupyter lab

Open any notebook and run all cells (most have CPU‑only fall‑backs, but GPUs are auto‑detected when available).

Requirements

Python ≥ 3.9

NumPy, Pandas, scikit‑learn, Matplotlib, Seaborn

PyTorch ≥ 2.1 (optional but recommended for GPU acceleration)

tqdm, SciPy, ipywidgets (for interactive sliders)

Install everything with:

pip install -r requirements.txt

A ready‑made requirements.txt is provided.

Data

Notebook‑specific CSVs live in the /data folder or are auto‑downloaded on first run.

Notebook

Dataset

Purpose

decision_tree.ipynb

train_data.csv, val_data.csv, test_data.csv

Transaction‑fraud detection

linear_regression.ipynb

dataset.csv

Couriers & delivery‑time prediction

knn*

embeddings.npy (auto‑generated demo)

Vector search benchmarks

Note: for privacy, raw proprietary data is not committed. The notebooks create synthetic or anonymised samples if the real files are absent.

Results

Approx‑KNN: IVF achieves ~95 % recall with 10× speed‑up over brute force on 100 k ×128‑D vectors; LSH reaches 80 % recall while fitting in RAM on laptops.

Decision Tree: AUC ≈ 0.89 on held‑out set; feature importance highlights high‑value receivers as prime fraud signals.

Linear Regression: R² ≈ 0.78, MAE ≈ 2.3 min; traffic & weather rank as top predictors.
