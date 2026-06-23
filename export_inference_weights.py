import os
import torch
import shutil

def export_weights():
    # Target directories setup
    base_out_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(base_out_dir, exist_ok=True)
    
    print("Starting generator weights extraction for 6 folds...")
    
    for fold in range(1, 7):
        fold_dir = os.path.join(base_out_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # 1. Export std2he generator weights
        # Full training checkpoints are stored under: output_cross_valid/SmplD1_5-45_{fold}/checkpoints/models/2000.pth
        std2he_src = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "output_cross_valid", 
            f"SmplD1_5-45_{fold}", 
            "checkpoints", 
            "models", 
            "2000.pth"
        )
        std2he_dst = os.path.join(fold_dir, "std2he_gene_a2b.pth")
        
        if os.path.exists(std2he_src):
            print(f"Processing std2he fold {fold} from: {std2he_src}")
            try:
                # Load with map_location='cpu' to avoid CUDA requirements
                checkpoint = torch.load(std2he_src, map_location='cpu')
                
                # Check for gene_a2b_state_dict
                if 'gene_a2b_state_dict' in checkpoint:
                    gene_state = checkpoint['gene_a2b_state_dict']
                    # Save only the generator state dict
                    torch.save({'gene_a2b_state_dict': gene_state}, std2he_dst)
                    orig_size_mb = os.path.getsize(std2he_src) / (1024 * 1024)
                    new_size_mb = os.path.getsize(std2he_dst) / (1024 * 1024)
                    print(f"  Saved to: {std2he_dst}")
                    print(f"  Size reduced from {orig_size_mb:.1f} MB to {new_size_mb:.1f} MB")
                else:
                    print(f"  Warning: 'gene_a2b_state_dict' not found in {std2he_src}. Copying as is.")
                    shutil.copy2(std2he_src, std2he_dst)
            except Exception as e:
                print(f"  Error processing std2he fold {fold}: {e}")
        else:
            print(f"  Source not found: {std2he_src}")
            
        # 2. Copy he2le generator weights
        # he2le weights are stored under: he2le/{fold}/1000.pth.gene_a2b
        he2le_src = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "he2le", 
            str(fold), 
            "1000.pth.gene_a2b"
        )
        he2le_dst = os.path.join(fold_dir, "he2le_gene_a2b.pth")
        
        if os.path.exists(he2le_src):
            print(f"Copying he2le fold {fold} from: {he2le_src}")
            try:
                shutil.copy2(he2le_src, he2le_dst)
                print(f"  Saved to: {he2le_dst}")
            except Exception as e:
                print(f"  Error copying he2le fold {fold}: {e}")
        else:
            print(f"  Source not found: {he2le_src}")

if __name__ == "__main__":
    export_weights()
