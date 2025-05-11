# DS_Tutorials

*Experiments in applied machine learning — from custom pipelines to EEG-based sleep stage classification.*

This repository contains two interconnected Data Science projects:

- **sleep_wave**: A full machine learning system for classifying sleep stages from EEG data, including feature engineering, sequence models, and custom sleep cycle segmentation.
- **pipeline_project**: A modular ML pipeline template for structured data, designed to streamline experimentation and evaluation.

These projects are built with reproducibility, extensibility, and hands-on learning in mind.

---

## Sleep_Wave: EEG-Based Sleep Stage Classification

`sleep_wave/` is a specialized deep learning pipeline for classifying sleep stages using EEG signals from the [PhysioNet Sleep-EDFx dataset](https://www.physionet.org/content/sleep-edfx/1.0.0/). It includes:

- EDF annotation parsing and alignment
- Bandpower, entropy, and temporal feature engineering
- Sequence-to-sequence LSTM modeling
- Hidden Markov Model (HMM) smoothing
- Custom cycle segmentation logic
- Leave-one-subject-out (LOSO) cross-validation evaluation

This project emphasizes both prediction performance and biological plausibility, reflecting how real human sleep cycles evolve over time.

---

## Pipeline_Project: Lightweight ML Workflow

This earlier project focuses on a reusable machine learning pipeline for structured datasets. It includes:

- Modular feature engineering framework
- Integrated hyperparameter tuning with `Optuna`
- Custom threshold tuning for imbalanced classification tasks
- Clean evaluation reports with metrics and plots

Many core ideas from this pipeline laid the foundation for the `sleep_wave` project.

---

## Repository Structure

```bash
.
├── sleep_wave/            # Sleep stage classification system
│   ├── data/              # Raw EDF files and annotations
│   ├── features/          # Feature builders (bandpower, entropy, etc.)
│   ├── models/            # LSTM + HMM model architectures
│   ├── experiments/       # Training and evaluation runners
│   └── utils/             # Helpers for preprocessing and cycles
├── pipeline_project/      # Standalone ML pipeline
├── notebooks/             # Exploratory Jupyter notebooks
├── requirements.txt       # Dependencies
└── README.md              # You're here!

