# EEWS-SpectrogramVAE-SAT
Variational Autoencoder (VAE)-based On-site Earthquake Early Warning System Framework using Accelerometer for Estimating Acceleration Response Spectra.

---

## Objective
This repository contains the source code for my undergraduate thesis at Universitas Indonesia. The project explores a deep learning-based EEWS framework to estimate **acceleration response spectra (SA(T))** from short-time seismic waveform data using spectrogram input. This method aims to support **On-site Earthquake Early Warning Systems (EEWS)** in Indonesia.

---

## Project Overview

Traditional EEWS frameworks rely on feature extraction and phase-picking to estimate earthquake intensity parameters. This project proposes a faster, end-to-end alternative using fixed-window **log-Mel spectrogram** inputs and **Variational Autoencoder (VAE)** models (CNN-DNN and CNN-LSTM) to reconstruct SA(T) values.

Key highlights:
- Input: 2–3s waveform spectrograms from accelerometer data (P-wave window).
- Output: Acceleration response spectra SA(T) for structural response analysis.
- Models: CNN-DNN VAE (more stable) and CNN-LSTM VAE (effective for short periods).
- Evaluation: Performed on both Indonesian (BMKG) and Italian (INSTANCE) datasets.

---

## Results Summary

- **CNN-DNN VAE (3s)**:
  - Average R² = 0.9460 on INSTANCE
  - Average R² = 0.8633 on BMKG
- **CNN-LSTM VAE (short-period T=0.01s on BMKG)**:
  - Average R² = 0.9749

---

## Project Structure

```
├── notebooks/           # Jupyter notebooks for training, evaluation, and visualization
├── py_utils/            # Preprocessing and plotting utilities
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Datasets

###  Public Dataset (included/sample-ready)
- [INSTANCE](https://www.pi.ingv.it/banche-dati/instance/):  
  *The Italian Seismic Dataset For Machine Learning*  
  > INSTANCE: The Italian Seismic Dataset For Machine Learning, Alberto Michelini, Spina Cianetti, Sonja Gaviano, Carlo Giunchi, Dario Jozinović & Valentino Lauciani. Published 2021 at Istituto Nazionale di Geofisica e Vulcanologia (INGV). [DOI: 10.13127/instance](https://doi.org/10.13127/instance)

### Restricted Dataset (not included)
- **BMKG Indonesian Earthquake Dataset**:  
  Access to this dataset is restricted. Please contact the BMKG for access.

---

## Disclaimer

This repository is intended for academic and educational purposes. The code provided for processing and modeling **INSTANCE data** is available. Any use of **BMKG data** requires proper authorization and is not included in this repository.

---
## Author

**Muhammad Fadli**  
Department of Physics, Universitas Indonesia
Email: muhammad.fadli12@ui.ac.id


## Citation

If you find this project helpful, please cite the following dataset and thesis:

> Fadli, M. (2025). *Kerangka Kerja Sistem Peringatan Dini Gempa Bumi di Tempat Menggunakan Sensor Akselerometer untuk Estimasi Spektrum Respons Percepatan SA(T) berbasis Variational Autoencoder (VAE)*. Universitas Indonesia.
