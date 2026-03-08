# VAE-Rumen-Microbiome-DeepLearning

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange) ![license MIT](https://img.shields.io/badge/license-MIT-green) ![status active](https://img.shields.io/badge/status-active-brightgreen)

> Variational Autoencoder (VAE) for unsupervised latent space modeling of rumen microbiome OTU profiles, with downstream regression to predict residual feed intake (RFI) in beef cattle.

## Biological Motivation

Rumen microbiome data is high-dimensional (thousands of OTUs) and sparse. Dimensionality reduction via VAE:
- Captures nonlinear community structure in a compressed latent space
- Enables visualization of microbial community clusters by host phenotype
- Provides latent features for downstream feed efficiency prediction

## Analysis Workflow

```
Raw OTU Count Matrix (samples x OTUs)
          |
          v
CLR Normalization + Zero Imputation
          |
          v
VAE Encoder: OTU -> mu, log_var (latent dim = 32)
          |
          v
Reparameterization: z = mu + eps * exp(0.5 * log_var)
          |
          v
VAE Decoder: z -> reconstructed OTU profile
          |
          v
Latent Space (z) -> Ridge Regression -> RFI prediction
          |
          v
PCA / UMAP of latent z -> Community visualization
```

## Model Architecture

| Component | Details |
|-----------|--------|
| Input dim | 1,200 OTUs (post-filtering) |
| Encoder layers | 1200 -> 512 -> 128 -> 32 (mu, log_var) |
| Decoder layers | 32 -> 128 -> 512 -> 1200 |
| Activation | ReLU (hidden), Sigmoid (output) |
| Loss | BCE reconstruction + KL divergence (beta=1) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| Epochs | 200 (early stopping patience=20) |
| Latent dim | 32 |

## Key Results

| Metric | Value |
|--------|-------|
| Reconstruction loss (test) | 0.043 |
| KL divergence (final) | 12.7 |
| RFI prediction R2 (latent -> Ridge) | 0.41 |
| UMAP clusters matching low/high RFI | 3 distinct clusters |
| Top latent dims correlated with RFI | z_7, z_12, z_19 |

## Repository Structure

```
VAE-Rumen-Microbiome-DeepLearning/
├── data/
│   ├── otu_table_filtered.csv      # CLR-normalized OTU matrix
│   └── sample_metadata.csv         # RFI, breed, age
├── models/
│   ├── vae_model.py                # VAE architecture (PyTorch)
│   └── train_vae.py                # Training loop
├── analysis/
│   ├── latent_regression.R         # Ridge regression on latent z
│   ├── umap_visualization.R        # UMAP of latent space
│   └── feature_importance.R        # Latent dim correlation with RFI
├── outputs/
│   ├── latent_embeddings.csv       # z vectors per sample
│   ├── umap_plot.pdf
│   └── model_weights.pt
└── README.md
```

## VAE Loss Function

```python
def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + beta * KLD
```

## Downstream Phenotype Prediction

Latent vectors (z) were extracted post-training and used as features:
- **Ridge Regression**: Predicts RFI from 32-dimensional latent space
- **Cross-validation**: 5-fold CV to avoid overfitting
- **Comparison**: VAE latent features vs raw OTU PCA components

## Dependencies

```bash
# Python
pip install torch torchvision numpy pandas scikit-learn umap-learn matplotlib seaborn

# R packages
library(glmnet)   # Ridge regression
library(umap)     # UMAP visualization
library(ggplot2)  # Plotting
library(tidyverse)
```

## Reproducibility

All models use `torch.manual_seed(2026)`. Training uses deterministic CUDA ops (`torch.use_deterministic_algorithms(True)`).

## Integration with Other Repos

This repo integrates downstream from:
- **Rumen-Microbiome-16S-Analysis** — OTU table generation
- **Rumen-mGWAS-Pipeline** — host genetic context

## Author

**Rushikesh Lagad** | PhD Researcher, Genomics & Bioinformatics | University of Arkansas
