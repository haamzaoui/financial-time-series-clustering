# Financial Time-Series Clustering

This repository contains the code and thesis material for a bachelor thesis on unsupervised pattern discovery in financial time series.

## Overview
This thesis investigates whether different unsupervised machine learning algorithms can identify recurring price patterns in financial time series data and evaluates the extent to which the resulting structures are well-separated, compact, and consistent across different methods, indicating structure beyond random assignment.

The analysis is based on adjusted daily closing prices of S&P 500 constituent stocks from 2000 to 2025.

The main methods compared are:

- K-Means
- Agglomerative Hierarchical Clustering (Ward linkage)
- DBSCAN

The evaluation focuses on:

- internal cluster quality
- cross-algorithm consistency
- qualitative comparison of cluster centers
- t-SNE visualization

## Data Pipeline

The workflow consists of:

1. downloading adjusted closing prices
2. segmenting each series into overlapping 50-day windows
3. normalizing each segment independently
4. drawing a shared sample for algorithm comparison
5. running clustering and evaluation

## Repository Structure

- `download/` – data download scripts
- `preprocessing/` – segmentation, normalization, and sampling
- `clustering/` – clustering scripts for K-Means, Hierarchical Clustering, and DBSCAN
- `results/` – generated outputs and figures
- thesis files – writing and documentation

## Reproducibility

For a reproducibility guide and instructions on how to run the full pipeline, see `DOCUMENTATION.md`.

## Notes

Large generated data files are excluded from version control. The repository is intended to store code, thesis text, and lightweight outputs only.

## Author

Mohamed Hamzaoui