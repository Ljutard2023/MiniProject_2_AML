# Mini-Project 2 — Variational Autoencoder Geometry
**Course:** Advanced Machine Learning (02460) — Technical University of Denmark  
**Topic:** Pull-back geodesics and ensemble VAE geometry on MNIST

---

## 📋 Project Overview

This project explores the geometry of the latent space learned by a Variational Autoencoder (VAE) trained on a subset of MNIST (digits 0, 1, 2 — 2048 samples).

**Part A:** Compute geodesics under the pull-back metric of a single VAE decoder.  
**Part B:** Use an ensemble of decoders to improve the robustness of geodesic distances, measured via the Coefficient of Variation (CoV).

---

## 📁 File Structure

```
AML_P2/
│
├── ensemble_vae.py          # Main script (all models + training + geodesics)
├── README.md
├── .gitignore
│
├── job_train.sh             # HPC job: train single VAE
├── job_geodesics.sh         # HPC job: compute Part A geodesics
├── job_ensemble.sh          # HPC job: train ensemble VAE
├── job_ensemble_geodesics.sh# HPC job: compute Part B geodesics
├── job_cov.sh               # HPC job: compute CoV (long!)
│
└── exp/                     # Output folder (created automatically)
    ├── model.pt                 # Trained single VAE
    ├── ensemble_model_3dec.pt   # Trained ensemble VAE
    ├── geodesics_partA.png      # Part A results
    ├── geodesics_partB_3dec.png # Part B results
    ├── cov_plot.png             # CoV plot
    ├── euclidean_dists.npy      # Raw euclidean distances
    └── geodesic_dists.npy       # Raw geodesic distances
```

---

## ⚙️ Installation

**Clone the repo:**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**Create and activate a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate       # Linux / Mac
# .venv\Scripts\activate        # Windows
```

**Install dependencies:**
```bash
pip install torch torchvision tqdm matplotlib numpy
```

> 💡 The MNIST dataset (~12MB) is downloaded automatically the first time you run the script.

---

## 🚀 How to Reproduce Results

### Option A — Local machine (CPU, slow)

Run each step sequentially:

```bash
# Step 1 — Train single VAE (Part A)
python ensemble_vae.py train --device cpu --experiment-folder exp

# Step 2 — Compute Part A geodesics
python ensemble_vae.py geodesics \
    --device cpu \
    --experiment-folder exp \
    --num-curves 25 \
    --num-t 20 \
    --geodesic-steps 500 \
    --geodesic-lr 1e-3

# Step 3 — Train ensemble VAE (Part B)
python ensemble_vae.py train_ensemble \
    --device cpu \
    --experiment-folder exp \
    --num-decoders 3

# Step 4 — Compute Part B geodesics
python ensemble_vae.py ensemble_geodesics \
    --device cpu \
    --experiment-folder exp \
    --num-decoders 3 \
    --num-curves 25 \
    --num-t 20 \
    --geodesic-steps 500 \
    --geodesic-lr 1e-3

# Step 5 — Compute CoV (VERY long on CPU — use GPU!)
python ensemble_vae.py compute_cov \
    --device cpu \
    --experiment-folder exp \
    --num-reruns 10 \
    --num-pairs-cov 10
```

---

### Option B — DTU HPC cluster (GPU, recommended)

**Setup:**
```bash
# Activate your venv on the login node
source /path/to/your/.venv/bin/activate
mkdir -p logs exp
```

**Submit jobs in order** (wait for each to finish before launching the next):

```bash
# Step 1 — Train single VAE
bsub < job_train.sh
bstat  # wait until DONE

# Step 2 — Part A geodesics
bsub < job_geodesics.sh
bstat

# Step 3 — Train ensemble VAE
bsub < job_ensemble.sh
bstat

# Step 4 — Part B geodesics
bsub < job_ensemble_geodesics.sh
bstat

# Step 5 — CoV (~2-3h on GPU, start early!)
bsub < job_cov.sh
bstat
```

**Monitor logs in real time:**
```bash
tail -f logs/<job_name>_<JOBID>.err
```

**Download results to your local machine:**
```bash
scp s<student_id>@hpclogin.dtu.dk:~/path/to/exp/*.png .
```

---

## 🔧 Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--experiment-folder` | `experiment` | Where to save/load models and plots |
| `--latent-dim` | `2` | Dimension of the latent space |
| `--epochs-per-decoder` | `50` | Training epochs per decoder |
| `--num-decoders` | `3` | Number of decoders in the ensemble |
| `--num-curves` | `25` | Number of geodesics to compute and plot |
| `--num-t` | `20` | Number of points along each curve |
| `--geodesic-steps` | `500` | Optimization steps per geodesic |
| `--geodesic-lr` | `1e-3` | Learning rate for geodesic optimizer |
| `--num-reruns` | `10` | Number of VAE retrainings for CoV |
| `--num-pairs-cov` | `10` | Number of point pairs for CoV |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` |

---

## 🧠 Code Structure

### Classes (provided)
| Class | Role |
|---|---|
| `GaussianPrior` | Standard normal prior over the latent space |
| `GaussianEncoder` | Convolutional encoder → latent distribution |
| `GaussianDecoder` | Transposed-conv decoder → image distribution |
| `VAE` | Combines prior + encoder + decoder, computes ELBO |

### Classes (added)
| Class | Role |
|---|---|
| `EnsembleVAE` | VAE with shared encoder and multiple independent decoders |

### Functions (added)
| Function | Role |
|---|---|
| `compute_curve_energy()` | Pull-back energy of a curve under a single decoder |
| `compute_ensemble_energy()` | Monte Carlo energy averaged over decoder ensemble |
| `compute_geodesic()` | Optimizes a curve to minimize energy (geodesic) |
| `curve_length_in_latent()` | Euclidean length of a curve (proxy for geodesic distance) |
| `train_ensemble_vae()` | Trains EnsembleVAE with frozen encoder strategy |
| `encode_dataset()` | Encodes full dataset, returns latent means + labels |
| `plot_latent_with_geodesics()` | Plots 2D latent space with geodesic curves |
| `plot_cov()` | Plots CoV vs. number of ensemble decoders |

---

## 📊 Results Summary

**Part A — Pull-back geodesics (single decoder)**  
Geodesics bend toward high-density regions of the latent space, particularly the Digit 1 cluster. This is expected behavior: the pull-back metric assigns low cost to regions where the decoder is well-trained.

**Part B — Ensemble geodesics (3 decoders)**  
Using multiple decoders produces more diverse geodesic paths and reduces sensitivity to any single decoder's geometry.

**Part B — CoV analysis**  
The geodesic CoV is consistently lower than the Euclidean CoV, confirming that pull-back distances are more robust to VAE retraining than simple Euclidean distances. The ensemble with 2 decoders achieves the best geodesic reliability.

---

## 👥 Authors
- Student 1 — s224403
- Student 2 — s224386
- Student 3 — s253050

DTU — Advanced Machine Learning (02460) — Spring 2026
