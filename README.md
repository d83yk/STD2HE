# Deep Generative Translation of Standard Images into Virtual High-Energy Images for Facilitating Dual-Energy Chest Radiography

Official implementation of the paper **"Deep Generative Translation of Standard Images into Virtual High-Energy Images for Facilitating Dual-Energy Chest Radiography"**.

This repository contains the source code, training/evaluation scripts, and model weights for the **STD2HE2LE** pipeline (coming soon). The pipeline is designed to reconstruct physical high-energy (HE) and low-energy (LE) domains from processed standard (STD) chest radiography images using a cascaded Pix2Pix cGAN model, allowing the synthesis of high-quality virtual bone-suppressed (BS) and bone-extracted (BE) images via dual-energy subtraction.

---

## Pipeline Overview

```
                   +------------------------+
                   |  Input Standard (STD)  | (0 to 1 normalized)
                   +-----------+------------+
                               |
                               v
                       [std2he Generator]
                               |
                               v
                 +-------------+------------+
                 |  Generated High-Energy   | (-1 to 1 normalized)
                 +-------------+------------+
                               |
                     +---------+---------+
                     |                   |
                     v                   v
             [he2le Generator]           |
                     |                   |
                     v                   v
       +-------------+------------+      |
       |    Generated Low-Energy  |      |
       +-------------+------------+      |
                     |                   |
                     +---------+---------+
                               |
                               v
                   [Dual-Energy Subtraction]
                               |
                      +--------+--------+
                      |                 |
                      v                 v
                 Soft Tissue (BS)   Bone (BE)
                   (rate=0.5)       (rate=1.0)
```

---

## Environment Setup

### Prerequisites
- Python >= 3.10
- PyTorch (with CUDA support recommended for fast inference)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/unet-gan-std2he2le.git
   cd unet-gan-std2he2le
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Model Weights
(It will be released soon.)
Because the generator weights exceed standard GitHub file size limits (~217 MB each), we host them using **GitHub Releases**. 

### Weight Folder Structure
The weights for the 6-fold cross-validation are organized as follows:
```
weights/
├── fold1/
│   ├── std2he_gene_a2b.pth
│   └── he2le_gene_a2b.pth
├── fold2/
│   ...
└── fold6/
    ├── std2he_gene_a2b.pth
    └── he2le_gene_a2b.pth
```

### Download Instructions
1. Go to the **Releases** page of this repository.
2. Download the weight assets for the desired folds.
3. Place the downloaded weight files into the corresponding `weights/fold{1-6}/` directories.

*Note: The generator weights have been optimized for inference by stripping optimizer and discriminator states, reducing the file size of `std2he` from 1.5 GB to 207 MB.*

---

## Usage

### 1. Inference (Generation of HE, LE, BS, BE)
To run the full cascaded generation on a single chest radiography image, run `predict.py`. The script accepts TIFF, DICOM, and standard formats (PNG/JPG):

```bash
python predict.py --input path/to/std_image.tif --output_dir output/ --fold 1
```

**Arguments:**
- `--input`: Path to the input standard (STD) chest X-ray image.
- `--output_dir`: Output directory to save the synthesized images.
- `--fold`: Cross-validation fold weights to load (choices: 1-6, default: 1).
- `--weights_dir`: Path to the directory where the fold weight folders are stored (default: `weights`).
- `--device`: Target device (e.g., `cuda` or `cpu`). Auto-detected by default.

**Outputs generated in the output directory:**
- `{input_basename}_HE.tif`: Reconstructed High-Energy image.
- `{input_basename}_LE.tif`: Reconstructed Low-Energy image.
- `{input_basename}_BS.tif`: Synthesized Soft Tissue (Bone-Suppressed) image.
- `{input_basename}_BE.tif`: Synthesized Bone (Bone-Extracted) image.

### 2. Evaluation / Testing
To compute evaluation metrics (MSE, PSNR, SSIM, LPIPS, DISTS, FID) on validation sets for all folds, run `test.py`:

```bash
python test.py
```

### 3. Model Training
To train the standard-to-high-energy generator model on your own dataset, prepare your paired dataset and run `train.py`:

```bash
python train.py --dataset_folder path/to/dataset --json_folder path/to/json --resume
```

---

## Citation

If you find this work or code useful for your research, please cite our paper:

```bibtex
@article{ueda2026std2he,
  title={Deep Generative Translation of Standard Images into Virtual High-Energy Images for Facilitating Dual-Energy Chest Radiography},
  author={Ueda, Y., Shimazaki, R., Seki, M., and Ishida, T.},
  journal={Journal of Imaging Informatics in Medicine},
  year={2026}
}
```
