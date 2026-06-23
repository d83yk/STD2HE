import os
import numpy as np
import torch
import csv
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import resize
from PIL import Image
from functools import partial
import warnings
warnings.simplefilter('ignore')
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from utils.dataset import PairedDataset, parse_args_options, save_args, save_image
from networks.calculator import des_image, norm11to01, norm01to11, lpips_compute, dists_compute

def compute_mse(gt_np, gen_np):
    return np.mean((gt_np - gen_np) ** 2)

def compute_psnr(gt_np, gen_np, data_range=1.0):
    return peak_signal_noise_ratio(gt_np, gen_np, data_range=data_range)

def compute_ssim(gt_np, gen_np, data_range=1.0):
    return structural_similarity(
        gt_np[:, :, 0], gen_np[:, :, 0],
        win_size=11, gaussian_weights=1.5,
        data_range=data_range
    )

def _init_tensor_stats():
    return {
        'sum': 0.0,
        'count': 0,
        'min': float('inf'),
        'max': float('-inf'),
    }

def _update_tensor_stats(stats, tensor):
    t = tensor.detach()
    stats['sum'] += float(t.sum().item())
    stats['count'] += int(t.numel())
    stats['min'] = min(stats['min'], float(t.min().item()))
    stats['max'] = max(stats['max'], float(t.max().item()))

def main(args):
    batch_size = 1

    # Adjust output folders
    args.test_folder = args.test_folder.replace("output_cross_valid", "output_cross_valid_eval")
    args.output_folder = args.output_folder.replace("output_cross_valid", "output_cross_valid_eval")

    os.makedirs(args.test_folder, exist_ok=True)
    args.testmodel_pth = os.path.join(args.train_model_folder, args.pth)
    print(f"Testing model checkpoint: {args.testmodel_pth}")
    
    dataset_test = PairedDataset(
        dataset_folder=args.dataset_folder, 
        json_folder=args.json_folder, 
        image_prep=args.test_image_prep, 
        split="val", # Uses validation set for evaluation
        mask_range=[0,0,100,100], 
        filename_A=args.filename_A, 
        filename_B=args.filename_B, 
        filename_C=args.filename_C, 
        args=args
    )
    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # Load generator models
    net_gene = args.generator
    epoch, best_epoch, *loss = net_gene.load_gene_a2b(args.testmodel_pth)

    net_he2le = args.he2le_generator
    net_he2le.load_gene_a2b(args.model_he2le_pth)
    net_he2le.set_eval()

    args.output_image_folder = os.path.join(args.test_folder, f'epoch={epoch:04d}')
    os.makedirs(args.output_image_folder, exist_ok=True)
    if args.is_save_image:
        for f in ['HE', 'LE', 'ST', 'BO']:
            os.makedirs(os.path.join(args.output_image_folder, f), exist_ok=True)

    keys_hl = ['h', 'l1', 'l2', 'st1', 'st2', 'bo1', 'bo2']
    keys_og = ['o', 'g1', 'g2']
    keys_gen = ['g1', 'g2']

    net_gene.set_eval()
    net_gene.set_lpips(args.device)
    net_gene.set_dists(args.device)
    net_gene = net_gene.to(args.device)
    
    save_args(args, os.path.join(args.output_folder, f"epoch={epoch:04d}_{os.path.basename(args.test_folder)}_args.json"))

    # Evaluation metric dictionaries
    metric_keys = {
        'dr':    ['std', 'he', 'le'],
        'mse':   keys_hl,
        'psnr':  keys_hl,
        'ssim':  keys_hl,
        'lpips': keys_hl,
        'dists': keys_hl,
    }
    
    fid_keys = {
        'fid_st': keys_gen,
        'fid_bo': keys_gen,
    }

    fid = {
        'fid_st': {k: FrechetInceptionDistance(normalize=False).to(args.device) for k in keys_gen},
        'fid_bo': {k: FrechetInceptionDistance(normalize=False).to(args.device) for k in keys_gen},
    }

    fid_input_stats = {
        'org_des_s': _init_tensor_stats(),
        'gen_des0_s': _init_tensor_stats(),
        'gen_des1_s': _init_tensor_stats(),
        'org_des_b': _init_tensor_stats(),
        'gen_des0_b': _init_tensor_stats(),
        'gen_des1_b': _init_tensor_stats(),
        'total': _init_tensor_stats(),
    }

    # Evaluation step
    with torch.no_grad():
        id_val = {'img_id': []}
        metrics_val = {name: {k: [] for k in keys} for name, keys in metric_keys.items()}
        fid_val = {name: {k: [] for k in keys_gen} for name in fid_keys.keys()}
        eval_val = id_val | metrics_val | fid_val

        for iter, batch in enumerate(sorted(dl_test)):
            img_id, org_mix, org_he, org_le, img_size = batch

            org_mix = org_mix.to(args.device, non_blocking=True)
            org_he = org_he.to(args.device, non_blocking=True)
            org_le = org_le.to(args.device, non_blocking=True)

            # Cascade forward pass
            gen_he = net_gene.gene_a2b.forward(org_mix)
            gen_le_1 = norm01to11(net_he2le.gene_a2b.forward(norm11to01(org_he)))
            gen_le_2 = norm01to11(net_he2le.gene_a2b.forward(norm11to01(gen_he)))

            dr_std_val = float((org_mix.max() - org_mix.min()).item())
            dr_he_val = float((org_he.max() - org_he.min()).item())
            dr_le_val = float((org_le.max() - org_le.min()).item())

            # Define des partial operators
            des_st = partial(des_image, rate=0.5, bit=16, log_tr_for_image_a=True, hist=False)
            des_bo = partial(des_image, rate=1.0, bit=16, log_tr_for_image_a=True, hist=False)

            # Subtraction images (ST / BO)
            orig_st = des_st(org_he, org_le).mul_(255).add_(0.5).clamp_(0, 255)
            gene_st_1 = des_st(org_he, gen_le_1).mul_(255).add_(0.5).clamp_(0, 255)
            gene_st_2 = des_st(gen_he, gen_le_2).mul_(255).add_(0.5).clamp_(0, 255)
            orig_bo = des_bo(org_he, org_le).mul_(255).add_(0.5).clamp_(0, 255)
            gene_bo_1 = des_bo(org_he, gen_le_1).mul_(255).add_(0.5).clamp_(0, 255)
            gene_bo_2 = des_bo(gen_he, gen_le_2).mul_(255).add_(0.5).clamp_(0, 255)

            # Numpy conversions scaled to [0, 1]
            he_gt_np = (org_he[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
            he_gen_np = (gen_he[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
            le_gt_np = (org_le[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
            le_gen1_np = (gen_le_1[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
            le_gen2_np = (gen_le_2[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0

            st_gt_np = orig_st[0].cpu().numpy().transpose(1, 2, 0) / 255.0
            st_gen1_np = gene_st_1[0].cpu().numpy().transpose(1, 2, 0) / 255.0
            st_gen2_np = gene_st_2[0].cpu().numpy().transpose(1, 2, 0) / 255.0
            bo_gt_np = orig_bo[0].cpu().numpy().transpose(1, 2, 0) / 255.0
            bo_gen1_np = gene_bo_1[0].cpu().numpy().transpose(1, 2, 0) / 255.0
            bo_gen2_np = gene_bo_2[0].cpu().numpy().transpose(1, 2, 0) / 255.0

            # Quality metrics calculations
            # MSE
            mse_he = compute_mse(he_gt_np, he_gen_np)
            mse_le1 = compute_mse(le_gt_np, le_gen1_np)
            mse_le2 = compute_mse(le_gt_np, le_gen2_np)
            mse_st1 = compute_mse(st_gt_np, st_gen1_np)
            mse_st2 = compute_mse(st_gt_np, st_gen2_np)
            mse_bo1 = compute_mse(bo_gt_np, bo_gen1_np)
            mse_bo2 = compute_mse(bo_gt_np, bo_gen2_np)

            # PSNR
            psnr_he = compute_psnr(he_gt_np, he_gen_np, 1.0)
            psnr_le1 = compute_psnr(le_gt_np, le_gen1_np, 1.0)
            psnr_le2 = compute_psnr(le_gt_np, le_gen2_np, 1.0)
            psnr_st1 = compute_psnr(st_gt_np, st_gen1_np, 1.0)
            psnr_st2 = compute_psnr(st_gt_np, st_gen2_np, 1.0)
            psnr_bo1 = compute_psnr(bo_gt_np, bo_gen1_np, 1.0)
            psnr_bo2 = compute_psnr(bo_gt_np, bo_gen2_np, 1.0)

            # SSIM
            ssim_he = compute_ssim(he_gt_np, he_gen_np, 1.0)
            ssim_le1 = compute_ssim(le_gt_np, le_gen1_np, 1.0)
            ssim_le2 = compute_ssim(le_gt_np, le_gen2_np, 1.0)
            ssim_st1 = compute_ssim(st_gt_np, st_gen1_np, 1.0)
            ssim_st2 = compute_ssim(st_gt_np, st_gen2_np, 1.0)
            ssim_bo1 = compute_ssim(bo_gt_np, bo_gen1_np, 1.0)
            ssim_bo2 = compute_ssim(bo_gt_np, bo_gen2_np, 1.0)

            # LPIPS & DISTS
            lpips_he = lpips_compute(org_he, gen_he, net_gene.net_lpips)
            lpips_le1 = lpips_compute(org_le, gen_le_1, net_gene.net_lpips)
            lpips_le2 = lpips_compute(org_le, gen_le_2, net_gene.net_lpips)

            dists_he = dists_compute(org_he, gen_he, net_gene.net_dists)
            dists_le1 = dists_compute(org_le, gen_le_1, net_gene.net_dists)
            dists_le2 = dists_compute(org_le, gen_le_2, net_gene.net_dists)

            t_orig_st_11 = (orig_st / 255.0) * 2.0 - 1.0
            t_gene_st_1_11 = (gene_st_1 / 255.0) * 2.0 - 1.0
            t_gene_st_2_11 = (gene_st_2 / 255.0) * 2.0 - 1.0
            t_orig_bo_11 = (orig_bo / 255.0) * 2.0 - 1.0
            t_gene_bo_1_11 = (gene_bo_1 / 255.0) * 2.0 - 1.0
            t_gene_bo_2_11 = (gene_bo_2 / 255.0) * 2.0 - 1.0

            lpips_st1 = lpips_compute(t_orig_st_11, t_gene_st_1_11, net_gene.net_lpips)
            lpips_st2 = lpips_compute(t_orig_st_11, t_gene_st_2_11, net_gene.net_lpips)
            lpips_bo1 = lpips_compute(t_orig_bo_11, t_gene_bo_1_11, net_gene.net_lpips)
            lpips_bo2 = lpips_compute(t_orig_bo_11, t_gene_bo_2_11, net_gene.net_lpips)

            dists_st1 = dists_compute(t_orig_st_11, t_gene_st_1_11, net_gene.net_dists)
            dists_st2 = dists_compute(t_orig_st_11, t_gene_st_2_11, net_gene.net_dists)
            dists_bo1 = dists_compute(t_orig_bo_11, t_gene_bo_1_11, net_gene.net_dists)
            dists_bo2 = dists_compute(t_orig_bo_11, t_gene_bo_2_11, net_gene.net_dists)

            # Record results
            eval_v = {
                'dr': {'std': dr_std_val, 'he': dr_he_val, 'le': dr_le_val},
                'mse': {'h': mse_he, 'l1': mse_le1, 'l2': mse_le2, 'st1': mse_st1, 'st2': mse_st2, 'bo1': mse_bo1, 'bo2': mse_bo2},
                'psnr': {'h': psnr_he, 'l1': psnr_le1, 'l2': psnr_le2, 'st1': psnr_st1, 'st2': psnr_st2, 'bo1': psnr_bo1, 'bo2': psnr_bo2},
                'ssim': {'h': ssim_he, 'l1': ssim_le1, 'l2': ssim_le2, 'st1': ssim_st1, 'st2': ssim_st2, 'bo1': ssim_bo1, 'bo2': ssim_bo2},
                'lpips': {'h': lpips_he, 'l1': lpips_le1, 'l2': lpips_le2, 'st1': lpips_st1, 'st2': lpips_st2, 'bo1': lpips_bo1, 'bo2': lpips_bo2},
                'dists': {'h': dists_he, 'l1': dists_le1, 'l2': dists_le2, 'st1': dists_st1, 'st2': dists_st2, 'bo1': dists_bo1, 'bo2': dists_bo2},
            }

            for k, v in eval_v.items():
                for r in metric_keys[k]:
                    eval_val[k][r].append(v[r])
            
            # Prepare tensors for FID
            gen_des0_b = des_image(org_he, gen_le_1, 1.0, 16, True)
            gen_des0_s = des_image(org_he, gen_le_1, 0.5, 16, True)
            gen_des1_b = des_image(gen_he, gen_le_2, 1.0, 16, True)
            gen_des1_s = des_image(gen_he, gen_le_2, 0.5, 16, True)
            org_des_b  = des_image(org_he, org_le, 1.0, 16, True)
            org_des_s  = des_image(org_he, org_le, 0.5, 16, True)

            for name, tens in [
                ('org_des_s', org_des_s), ('gen_des0_s', gen_des0_s), ('gen_des1_s', gen_des1_s),
                ('org_des_b', org_des_b), ('gen_des0_b', gen_des0_b), ('gen_des1_b', gen_des1_b),
            ]:
                _update_tensor_stats(fid_input_stats[name], tens)
                _update_tensor_stats(fid_input_stats['total'], tens)

            org_des_s_u8 = org_des_s.repeat(1,3,1,1).clamp(0, 255).to(torch.uint8)
            gen_des0_s_u8 = gen_des0_s.repeat(1,3,1,1).clamp(0, 255).to(torch.uint8)
            gen_des1_s_u8 = gen_des1_s.repeat(1,3,1,1).clamp(0, 255).to(torch.uint8)
            org_des_b_u8 = org_des_b.repeat(1,3,1,1).clamp(0, 255).to(torch.uint8)
            gen_des0_b_u8 = gen_des0_b.repeat(1,3,1,1).clamp(0, 255).to(torch.uint8)
            gen_des1_b_u8 = gen_des1_b.repeat(1,3,1,1).clamp(0, 255).to(torch.uint8)

            fid['fid_st']['g1'].update(org_des_s_u8, real=True)
            fid['fid_st']['g1'].update(gen_des0_s_u8, real=False)
            fid['fid_st']['g2'].update(org_des_s_u8, real=True)
            fid['fid_st']['g2'].update(gen_des1_s_u8, real=False)

            fid['fid_bo']['g1'].update(org_des_b_u8, real=True)
            fid['fid_bo']['g1'].update(gen_des0_b_u8, real=False)
            fid['fid_bo']['g2'].update(org_des_b_u8, real=True)
            fid['fid_bo']['g2'].update(gen_des1_b_u8, real=False)

            for _id, _mx, _oh, _ol, _gh, _g0l, _g1l, _g0b, _g0s, _g1b, _g1s, _ob, _os in zip(
                img_id, org_mix, norm11to01(org_he), norm11to01(org_le), norm11to01(gen_he), norm11to01(gen_le_1), norm11to01(gen_le_2),
                gen_des0_b, gen_des0_s, gen_des1_b, gen_des1_s, org_des_b, org_des_s
            ):
                if args.is_save_image:
                    save_image(torch.cat([_oh, _gh], dim=2), os.path.join(args.output_image_folder, "HE", f"{_id}.tif"), dtype=np.uint16, dyn_range=2**12-1)
                    save_image(_g0l, os.path.join(args.output_image_folder, "LE", f"{_id}_AIDES.tif"), dtype=np.uint16, dyn_range=2**12-1)
                    save_image(_g0s, os.path.join(args.output_image_folder, "ST", f"{_id}_AIDES.tif"), dtype=np.uint16, dyn_range=2**12-1)
                    save_image(_g0b, os.path.join(args.output_image_folder, "BO", f"{_id}_AIDES.tif"), dtype=np.uint16, dyn_range=2**12-1)
                    save_image(torch.cat([_ol, _g1l], dim=2), os.path.join(args.output_image_folder, "LE", f"{_id}.tif"), dtype=np.uint16, dyn_range=2**12-1)
                    save_image(torch.cat([_os, _g1s], dim=2), os.path.join(args.output_image_folder, "ST", f"{_id}.tif"), dtype=np.uint16, dyn_range=2**12-1)
                    save_image(torch.cat([_ob, _g1b], dim=2), os.path.join(args.output_image_folder, "BO", f"{_id}.tif"), dtype=np.uint16, dyn_range=2**12-1)

                eval_val['img_id'].append(_id)

        # Compute FID
        for k in fid_keys.keys():
            for g in keys_gen:
                eval_val[k][g].append(fid[k][g].compute().cpu().float())

        # Display mean summary
        print("====== Evaluation Summary ======")
        for m in metric_keys.keys():
            print(f"  {m.upper()}: ", end='')
            for r in metric_keys[m]:
                mean_v = np.mean(eval_val[m][r])
                print(f"{r}={mean_v:.4f} ", end='')
            print()
        
        for m in fid_keys.keys():
            print(f"  {m.upper()}: ", end='')
            for g in keys_gen:
                print(f"{g}={eval_val[m][g][-1]:.4f} ", end='')
            print()

        # Save to csv
        csv_path = os.path.join(args.output_image_folder, "scores.csv")
        with open(csv_path, 'w', newline="") as f:
            writer = csv.writer(f)
            header = ["img_id"]
            for m in metric_keys:
                for r in metric_keys[m]:
                    header.append(f"{m}-{r}")
            for m in fid_keys:
                for r in fid_keys[m]:
                    header.append(f"{m}-{r}")
            writer.writerow(header)

            num_samples = len(eval_val["img_id"])
            for i in range(num_samples):
                row = [eval_val["img_id"][i]]
                for m in metric_keys:
                    for r in metric_keys[m]:
                        val = eval_val[m][r][i]
                        row.append(val.item() if isinstance(val, torch.Tensor) else val)
                for m in fid_keys:
                    for r in fid_keys[m]:
                        row.append(eval_val[m][r][0].item())
                writer.writerow(row)
        print(f"Scores exported to: {csv_path}")

if __name__ == "__main__":
    name_prefix = 'SmplD1'
    min_w = 5
    max_w = 45
    cross_val_num = [1, 2, 3, 4, 5, 6]
    for _c in cross_val_num:
        print(f"\n--- Evaluating Fold {_c} ---")
        args = parse_args_options(
            is_train=False, 
            name_prefix=name_prefix, 
            cross_val_num=_c,
            min_w=min_w,
            max_w=max_w,
            roi_eval="sum_annot",
            pth="2000.pth",
            is_save_image=False,
        )
        # Point to the exported weights
        args.train_model_folder = f"./weights/fold{_c}/"
        args.pth = "std2he_gene_a2b.pth"
        main(args)
