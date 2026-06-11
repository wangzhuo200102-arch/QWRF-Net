# Code Availability

**Name of code/library:** QWRF-Net (Quantum-Wavelet Framework with Rectified Flow for Short-Term Precipitation Nowcasting)

**Contact:** Chaorong Li, lichaorong88@163.com

**Hardware requirements:** A machine with NVIDIA GPUs is recommended. The experiments reported in this study were conducted using NVIDIA A6000 GPUs. A CUDA-compatible GPU is recommended for training and evaluation.

**Program language:** Python

**Software required:** Python 3.9, PyTorch 2.1.0+cu121, PennyLane 0.34.0, pytorch_wavelets 1.3.0, NumPy 1.26.2, CUDA 12.1, and related scientific-computing libraries, including SciPy 1.11.4 and Matplotlib 3.8.2.

**Program size:** The source code is lightweight and consists of the QWRF-Net model implementation, rectified-flow module, time-sampling utilities, dataset utilities, metric functions, visualization utilities, and running scripts for training, evaluation, and inference.

**Source code repository:** https://github.com/wangzhuo200102-arch/QWRF-Net

**Pretrained model checkpoints:** The pretrained weights are not stored directly in the GitHub repository because of file-size limitations. They are available at the following links:

| Dataset | File | Download link | Extraction code |
|---|---|---|---|
| SEVIR | `best_model_sevir.pth` | https://pan.baidu.com/s/1YECdIfW0pqc_k082cyYhmQ?pwd=qwrf | `qwrf` |
| KNMI | `best_model_knmi.pth` | https://pan.baidu.com/s/1y4PCdojqvgRON7wnX3Mm6A?pwd=qwrf | `qwrf` |

After downloading, place the checkpoints in:

```text
checkpoints/best_model_sevir.pth
checkpoints/best_model_knmi.pth
```

**Datasets:** The datasets are not included in the repository. Users should download SEVIR and KNMI data from their official data providers and follow the corresponding data-access policies and license terms.

**License:** The source code is released under the MIT License.

**Main executable scripts:**

```text
scripts/prepare_sevir_index.py
scripts/train.py
scripts/evaluate.py
scripts/inference.py
```

