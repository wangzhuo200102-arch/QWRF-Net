# QWRF-Net

Official implementation of **QWRF-Net: A Hybrid Quantum-Wavelet Rectified Flow Network for Precipitation Nowcasting**.

QWRF-Net is a conditional generative precipitation nowcasting model that combines:

- **Wavelet-based multi-scale feature decomposition**
- **Quantum-inspired nonlinear feature transformation**
- **Rectified-flow-based non-autoregressive sequence generation**

The model is designed for short-term precipitation nowcasting and is evaluated on datasets such as **KNMI** and **SEVIR**.

---

## Repository Structure

This repository currently contains the core implementation files:

```text
QWRF-Net/
├── README.md
├── __init__.py
├── mainmd.py
├── model.py
├── rectified_flow.py
└── time_sampler.py


Pretrained Weights

The pretrained checkpoints can be downloaded from Baidu Netdisk:

Link: https://pan.baidu.com/s/1Psa0bXa54Cl2aIPKIqQJog?pwd=qwrf

Extraction code: qwrf
