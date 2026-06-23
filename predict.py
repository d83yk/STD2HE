import os
import argparse
import torch
import numpy as np
import tifffile as tiff
from PIL import Image
import torchvision.transforms as transforms
import pydicom

from networks.generators import Single_Generator, Unet8BN, Unet8
from networks.calculator import des_image, norm11to01, norm01to11
from utils.dataset import save_image

def load_input_image(path):
    """Loads a chest X-ray image (DICOM, TIFF, PNG/JPG) and normalizes it to [0, 1]."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    # Read image based on extension
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.dcm', '.dicom']:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
            img = float(ds.RescaleIntercept) + img * float(ds.RescaleSlope)
        
        # Normalize window values to [0, 1]
        min_w = np.percentile(img, 0)
        max_w = np.percentile(img, 100)
        ww = max_w - min_w + 1e-6
        img = np.clip(img - min_w, 0, ww) / ww
        pil_img = Image.fromarray(img.astype(np.float32)).convert('F')
    elif ext in ['.tif', '.tiff']:
        img = tiff.imread(path).astype(np.float64)
        min_w = np.percentile(img, 0)
        max_w = np.percentile(img, 100)
        ww = max_w - min_w + 1e-6
        img = np.clip(img - min_w, 0, ww) / ww
        pil_img = Image.fromarray(img).convert('F')
    else:
        # Standard formats (PNG, JPG, etc.)
        img = Image.open(path).convert('L')
        img_np = np.array(img).astype(np.float32) / 255.0
        pil_img = Image.fromarray(img_np).convert('F')

    # Resize to 1024x1024 as required by the model and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    tensor = transform(pil_img).unsqueeze(0)  # Shape: [1, 1, 1024, 1024]
    return tensor

def predict(args):
    # Determine device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Set paths to model weights based on selected fold
    weights_fold_dir = os.path.join(args.weights_dir, f"fold{args.fold}")
    std2he_path = os.path.join(weights_fold_dir, "std2he_gene_a2b.pth")
    he2le_path = os.path.join(weights_fold_dir, "he2le_gene_a2b.pth")

    if not os.path.exists(std2he_path) or not os.path.exists(he2le_path):
        raise FileNotFoundError(
            f"Weights not found for fold {args.fold} in {weights_fold_dir}.\n"
            "Please run 'export_inference_weights.py' first to extract them, or download them to the weights directory."
        )

    # Initialize generators
    print("Loading STD2HE model...")
    std2he = Single_Generator(
        Unet8BN, 
        model_args=dict(in_channels=1, out_channels=1, device=device), 
        is_train=False
    ).to(device)
    std2he.load_gene_a2b(std2he_path)
    std2he.set_eval()

    print("Loading HE2LE model...")
    he2le = Single_Generator(
        Unet8, 
        model_args=dict(in_channels=1, device=device), 
        is_train=False
    ).to(device)
    he2le.load_gene_a2b(he2le_path)
    he2le.set_eval()

    # Load input standard (STD) image
    print(f"Loading input standard (STD) image: {args.input}")
    std_tensor = load_input_image(args.input).to(device)

    # Output folder setup
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    with torch.no_grad():
        print("Running cascaded inference...")
        # 1. Standard (STD) -> High-Energy (HE)
        # STD input is in [0, 1], HE output is in [-1, 1]
        gen_he = std2he.gene_a2b(std_tensor)

        # 2. High-Energy (HE) -> Low-Energy (LE)
        # HE is normalized to [0, 1] for he2le input, output LE is mapped back to [-1, 1]
        gen_le = norm01to11(he2le.gene_a2b(norm11to01(gen_he)))

        # 3. Compute Dual-Energy Subtraction (DES) images
        # BS (Bone Suppression / Soft Tissue): rate = 0.5
        # BE (Bone Extraction / Bone): rate = 1.0
        print("Computing Bone-Suppressed (BS) and Bone-Extracted (BE) subtraction images...")
        gen_st = des_image(gen_he, gen_le, rate=0.5, bit=16, log_tr_for_image_a=True, hist=False)
        gen_bo = des_image(gen_he, gen_le, rate=1.0, bit=16, log_tr_for_image_a=True, hist=False)

        # Scale des images to [0, 255] and clamp as done in training
        gen_st_u8 = gen_st.mul(255).add_(0.5).clamp(0, 255)
        gen_bo_u8 = gen_bo.mul(255).add_(0.5).clamp(0, 255)

        # 4. Save results
        # Save generated HE and LE (mapped from [-1, 1] to [0, 1] for saving as tiff)
        he_save = norm11to01(gen_he)[0]
        le_save = norm11to01(gen_le)[0]
        
        save_image(he_save, os.path.join(args.output_dir, f"{base_name}_HE.tif"), dyn_range=2**12-1, dtype=np.uint16)
        save_image(le_save, os.path.join(args.output_dir, f"{base_name}_LE.tif"), dyn_range=2**12-1, dtype=np.uint16)
        save_image(gen_st_u8[0], os.path.join(args.output_dir, f"{base_name}_BS.tif"), dyn_range=1.0, dtype=np.uint8)
        save_image(gen_bo_u8[0], os.path.join(args.output_dir, f"{base_name}_BE.tif"), dyn_range=1.0, dtype=np.uint8)

    print(f"Inference complete! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STD2HE2LE Cascaded Generation & Dual-Energy Subtraction Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input standard chest X-ray image (TIFF, DICOM, or PNG/JPG)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save generated output images")
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 7), help="Cross-validation fold model weights to load (1-6)")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory where model weights folder (e.g. fold1/) are located")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu'). Auto-detected if not specified.")
    
    args = parser.parse_args()
    predict(args)
