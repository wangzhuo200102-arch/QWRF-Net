# QWRF-Net

Official implementation of **QWRF-Net: A Quantum-Wavelet Framework with Rectified Flow for Short-Term Precipitation Nowcasting**.

QWRF-Net is a conditional generative precipitation nowcasting model. Given recent radar precipitation frames, it predicts future precipitation fields using a quantum-wavelet feature transformation module and a rectified-flow generation framework.

## Code Availability

**Name of code/library:** QWRF-Net (Quantum-Wavelet Framework with Rectified Flow for Short-Term Precipitation Nowcasting)

**Contact:** Chaorong Li, lichaorong88@163.com

**Hardware requirements:** A machine with NVIDIA GPUs is recommended. The experiments reported in this study were conducted using NVIDIA A6000 GPUs. A CUDA-compatible GPU is recommended for training and evaluation.

**Program language:** Python

**Software required:** Python 3.9, PyTorch 2.1.0+cu121, PennyLane 0.34.0, pytorch_wavelets 1.3.0, NumPy 1.26.2, CUDA 12.1, SciPy 1.11.4, and Matplotlib 3.8.2.

**Program size:** The source code is lightweight and consists of the QWRF-Net model implementation, rectified-flow module, time-sampling utilities, dataset utilities, metric functions, visualization utilities, and running scripts.

**Source code repository:** https://github.com/wangzhuo200102-arch/QWRF-Net

See [CODE_AVAILABILITY.md](CODE_AVAILABILITY.md) for checkpoint links, dataset notes, and executable scripts.

## Main Features

- Conditional precipitation nowcasting from 6 input frames to 12 future frames.
- QWRF-Net model architecture with multi-scale attention blocks.
- Hybrid quantum-wavelet bottleneck for nonlinear multi-scale feature transformation.
- Rectified-flow scheduler and sampler.
- SEVIR-compatible dataset indexing and loading utilities.
- Training, evaluation, inference, and visualization scripts.

## Repository Structure

```text
QWRF-Net/
|-- README.md
|-- qwrfnet/
|   |-- __init__.py
|   |-- model.py
|   |-- rectified_flow.py
|   |-- sampler.py
|   |-- time_sampler.py
|   |-- dataset.py
|   |-- metrics.py
|   `-- visualization.py
`-- scripts/
|   |-- prepare_sevir_index.py
|   |-- train.py
|   |-- evaluate.py
|   `-- inference.py
```

## Installation

Create a Python environment and install the required packages.

Tested software versions:

```text
Python: 3.9
PyTorch: 2.1.0+cu121
TorchVision: 0.16.0
CUDA: 12.1
NumPy: 1.26.2
SciPy: 1.11.4
Matplotlib: 3.8.2
PennyLane: 0.34.0
pytorch-wavelets: 1.3.0
PyYAML: 6.0.1
h5py: 3.10.0
scikit-learn: 1.3.2
einops: 0.7.0
```

```bash
git clone https://github.com/YOUR_USERNAME/QWRF-Net.git
cd QWRF-Net
pip install -r requirements.txt
```

Alternatively, create the conda environment:

```bash
conda env create -f environment.yml
conda activate qwrfnet
```

`pennylane` and `pytorch-wavelets` are required for the full quantum-wavelet model.

## Data Format

QWRF-Net uses 6 previous precipitation frames as input and predicts 12 future frames.

```text
condition: [B, 6, H, W]
target:    [B, 12, H, W]
output:    [B, 12, H, W]
```

Values are normalized by the maximum precipitation/VIL value used in the corresponding dataset.

## Dataset Access

The datasets used by this project are not included in this repository. Users must download the datasets from their official providers and follow the corresponding data license and access policies.

This repository only provides:

- model source code
- training and evaluation scripts
- data format descriptions
- sample indexing and loading utilities
- pretrained model checkpoint links

## SEVIR Data Preparation

Download SEVIR from the official SEVIR data provider before running the experiments. After downloading, organize the VIL files as:

```text
/path/to/sevir/vil/
|-- 2017/
|-- 2018/
`-- 2019/
```

Generate a valid sample index:

```bash
python scripts/prepare_sevir_index.py \
  --data-path /path/to/sevir/vil \
  --output data/sevir_valid_samples.json
```

The loader expects HDF5 files containing a `vil` variable. The first 6 frames are used as conditioning input, and the following 12 frames are used as prediction targets.

## KNMI Data Preparation

Download the KNMI precipitation nowcasting dataset from the official KNMI data source before running KNMI experiments. Because access rules and file organization may vary by release, follow the official KNMI download and license instructions.

After downloading and preprocessing, provide the prepared sample directory to the training or evaluation script:

```bash
python scripts/evaluate.py \
  --config configs/knmi_qwrfnet.yaml \
  --data-dir /path/to/prepared/knmi_samples \
  --checkpoint checkpoints/best_model_knmi.pth \
  --output outputs/evaluation_knmi
```

## Training

Train QWRF-Net after preparing the dataset and configuration:

```bash
python scripts/train.py \
  --config configs/sevir_qwrfnet.yaml \
  --data-dir /path/to/prepared/sevir_or_knmi_samples \
  --output outputs/qwrfnet
```

The checkpoint is saved as:

```text
outputs/qwrfnet/best_model.pth
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/sevir_qwrfnet.yaml \
  --data-dir /path/to/prepared/sevir_or_knmi_samples \
  --checkpoint checkpoints/best_model_sevir.pth \
  --output outputs/evaluation_sevir
```

Evaluation outputs include:

```text
outputs/evaluation_sevir/metrics.json
outputs/evaluation_sevir/sample_preview.png
```

The evaluation script reports:

- MSE
- RMSE
- MAE
- SSIM
- CSI
- HSS

## Inference

Prepare an input file with shape:

```text
[6, H, W]
```

Then run:

```bash
python scripts/inference.py \
  --config configs/sevir_qwrfnet.yaml \
  --input /path/to/input.npy \
  --checkpoint checkpoints/best_model_sevir.pth \
  --output outputs/inference
```

The output files are:

```text
outputs/inference/prediction.npy
outputs/inference/prediction_preview.png
```

## Pretrained Weights

Pretrained weights are not stored directly in this repository because the files are too large for GitHub.

Current checkpoints:

| Dataset | File | Baidu Netdisk | Extraction Code |
|---|---|---|---|
| SEVIR | `best_model_sevir.pth` | https://pan.baidu.com/s/1YECdIfW0pqc_k082cyYhmQ?pwd=qwrf | `qwrf` |
| KNMI | `best_model_knmi.pth` | https://pan.baidu.com/s/1y4PCdojqvgRON7wnX3Mm6A?pwd=qwrf | `qwrf` |

After downloading, place the checkpoints at:

```text
checkpoints/best_model_sevir.pth
checkpoints/best_model_knmi.pth
```

Evaluate with the SEVIR checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/sevir_qwrfnet.yaml \
  --data-dir /path/to/prepared/sevir_samples \
  --checkpoint checkpoints/best_model_sevir.pth \
  --output outputs/evaluation_sevir
```

Evaluate with the KNMI checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/knmi_qwrfnet.yaml \
  --data-dir /path/to/prepared/knmi_samples \
  --checkpoint checkpoints/best_model_knmi.pth \
  --output outputs/evaluation_knmi
```

Recommended public hosting options for international accessibility:

- Zenodo
- Hugging Face
- Figshare
- OSF
- GitHub Releases, if the file size is acceptable

Verify checkpoint files with SHA256.

Linux/macOS:

```bash
sha256sum checkpoints/best_model_sevir.pth
sha256sum checkpoints/best_model_knmi.pth
```

Windows:

```powershell
certutil -hashfile checkpoints/best_model_sevir.pth SHA256
certutil -hashfile checkpoints/best_model_knmi.pth SHA256
```

## Main Python Modules

`qwrfnet/model.py`  
Defines the QWRF-Net architecture, including multi-scale attention blocks, conditional encoder, quantum-wavelet bottleneck, and output head.

`qwrfnet/rectified_flow.py`  
Implements the rectified-flow training scheduler and velocity-matching loss.

`qwrfnet/sampler.py`  
Provides the `RFLOW2D` sampler for 2D precipitation sequence generation.

`qwrfnet/time_sampler.py`  
Implements uniform and logit-normal time-step sampling.

`qwrfnet/dataset.py`  
Provides SEVIR VIL indexing and dataset loading utilities.

`qwrfnet/metrics.py`  
Implements continuous and threshold-based nowcasting metrics.

`qwrfnet/visualization.py`  
Saves ground-truth and prediction comparison figures.



## Reproducibility Notes

For journal review and reuse, the public repository should include:

- `LICENSE`
- `requirements.txt` or `environment.yml`
- public checkpoint links
- SHA256 checksums for external checkpoints
- dataset access instructions pointing to official data providers
- exact training and evaluation settings
- all code comments in English
