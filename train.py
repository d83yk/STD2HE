import os
import torch
import shutil
from torch.amp.grad_scaler import GradScaler
from statistics import mean
from glob import glob
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid
import numpy as np
import warnings
warnings.simplefilter('ignore')

from utils.dataset import PairedDataset, parse_args_options, save_args, set_seed, save_image
from networks.calculator import des_image, norm11to01

def main(args):
    scaler_g = scaler_d = GradScaler(device=args.device, enabled=args.use_amp)
    set_seed(args.seed)

    save_args(args, os.path.join(args.output_folder, "train_args.json"))

    os.makedirs(args.eval_folder, exist_ok=True)
    os.makedirs(args.ckpt_folder, exist_ok=True)
    os.makedirs(args.train_model_folder, exist_ok=True)

    currentmodel_cpt = os.path.join(args.train_model_folder, 'latest.pth')
    bestmodel_cpt = os.path.join(args.train_model_folder, 'best.pth')
    loss_txt = os.path.join(args.ckpt_folder, 'loss.txt')

    if not os.path.exists(args.num_training_epochs_txt[0]):
        with open(args.num_training_epochs_txt[0], 'w') as f:
            print(args.num_training_epochs, file=f)

    # Dataloaders
    dataset_train = PairedDataset(
        dataset_folder=args.dataset_folder, 
        json_folder=args.json_folder, 
        image_prep=args.train_image_prep, 
        split="train", 
        mask_range=args.mask_range, 
        filename_A=args.filename_A, 
        filename_B=args.filename_B
    )
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    
    dataset_valid = PairedDataset(
        dataset_folder=args.dataset_folder, 
        json_folder=args.json_folder, 
        image_prep=args.val_image_prep, 
        split="val", 
        mask_range=args.mask_range, 
        filename_A=args.filename_A, 
        filename_B=args.filename_B, 
        filename_C=args.filename_C, 
        args=args
    )
    dl_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=0)

    saveimage_train = dataset_train.img_ids[0]
    os.makedirs(os.path.join(args.ckpt_folder, saveimage_train), exist_ok=True)

    # Initialize generators & discriminators
    net_gene = args.generator
    net_gene.print_networks(name='net_generator', save_dir=args.train_model_folder)
    net_gene.set_lpips(args.device)
    net_gene.set_dists(args.device)
    net_gene.initialize_networks(init_type='xavier_uniform', device=args.device)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_disc = args.discriminator
    net_disc.print_networks(name="net_discriminator", save_dir=args.train_model_folder)
    net_disc.initialize_networks(init_type='xavier_uniform', device=args.device)

    # Load pre-trained he2le generator for validation
    net_he2le = args.he2le_generator
    net_he2le.load_gene_a2b(args.model_he2le_pth)
    net_he2le.set_eval()

    # Resume training if checkpoint is specified
    continue_train = getattr(args, 'resume', False)
    next_epoch = best_epoch = 0
    loss_gene_best = float('inf')

    if continue_train and os.path.exists(currentmodel_cpt):
        print(f"Resuming training from: {currentmodel_cpt}")
        next_epoch, best_epoch, loss_gene_epochs, loss_gene_best = net_gene.load_networks(currentmodel_cpt, load_optim=True)
        _, loss_disc = net_disc.load_networks(currentmodel_cpt, load_optim=True)
        print('\r epoch: [%3d/%3d] LossG: %.3f (Best: %.3f@%3d) LossD: %.3f' % (
            next_epoch, args.num_training_epochs, loss_gene_epochs, loss_gene_best, best_epoch, loss_disc
        ))
        next_epoch += 1
    else:
        with open(loss_txt, 'w') as f:
            print('epoch,lossG,lossD,SSIM,PSNR,LPIPS,DISTS', file=f)

    total_iters = next_epoch * len(dataset_train)

    def loss_x_lambda(_loss, _lambda):
        return [loss_ * lambda_ for (loss_, lambda_) in zip(_loss, _lambda)]

    for epoch in range(next_epoch, args.num_training_epochs + 1):
        loss_gene_fwd_epochs = loss_gene_adv_epochs = 0.
        loss_disc_adv_org_epochs = loss_disc_adv_gen_epochs = loss_disc_adv_mix_epochs = 0.

        # Training Step
        net_gene.set_train()
        net_disc.set_train()
        for _iter, batch in enumerate(dl_train):
            iter = _iter + 1
            img_ids, img_mixs, img_highs, masks, _, _ = batch

            img_mixs = img_mixs.to(args.device)
            img_highs = img_highs.to(args.device)
            masks = masks.to(args.device)
            total_iters += img_mixs.shape[0]

            # 1. Update Discriminator
            net_disc.set_requires_grad(True)
            if iter % args.gradient_accumulation_steps == 1:
                net_disc.optimizer.zero_grad()

            with torch.autocast(device_type=str(args.device), enabled=args.use_amp, dtype=args.mixed_precision):
                generate_b = net_gene.gene_a2b.forward(img_mixs)
                loss_disc_adv_gen_b, loss_disc_cr_gen = net_disc.get_loss_for_D_gen(net_disc.disc_b, generate_b)
                loss_disc_adv_org_b, loss_disc_cr_org = net_disc.get_loss_for_D_org(net_disc.disc_b, img_highs)
                loss_disc_adv_mix_b, loss_disc_cr_mix = net_disc.get_loss_for_D_gen(net_disc.disc_b, net_disc.cutmix_image(img_highs, generate_b, masks))

                loss_disc_adv = sum([loss_disc_adv_org_b * args.lambda_gan[0], loss_disc_adv_gen_b * args.lambda_gan[0], loss_disc_adv_mix_b * args.lambda_gan[0]])
                loss_disc_cr = sum([loss_disc_cr_org * args.lambda_gan[0], loss_disc_cr_gen * args.lambda_gan[0], loss_disc_cr_mix * args.lambda_gan[0]])

            loss_disc = (loss_disc_adv + loss_disc_cr) * args.lambda_disc
            loss_disc_scaled = loss_disc / args.gradient_accumulation_steps
            scaler_d.scale(loss_disc_scaled).backward()

            if iter % args.gradient_accumulation_steps == 0:
                scaler_d.step(net_disc.optimizer)
                scaler_d.update()            

            # 2. Update Generator
            net_disc.set_requires_grad(False)
            if iter % args.gradient_accumulation_steps == 1:
                net_gene.optimizer.zero_grad()

            with torch.autocast(device_type=str(args.device), enabled=args.use_amp, dtype=args.mixed_precision):
                generate_b = net_gene.gene_a2b.forward(img_mixs)
                loss_gene_adv_b = net_disc.get_loss_for_G(net_disc.disc_b, generate_b)
                loss_gene_adv = loss_gene_adv_b * args.lambda_gan[0]
                
                loss_gene_fwd_each = net_gene.get_loss(img_mixs, img_highs)
                loss_gene_fwd_each = loss_x_lambda(loss_gene_fwd_each, [args.lambda_l1, args.lambda_percept])
                loss_gene_fwd = sum(loss_gene_fwd_each)

            loss_gene = loss_gene_adv + loss_gene_fwd
            loss_gene_scaled = loss_gene / args.gradient_accumulation_steps
            scaler_g.scale(loss_gene_scaled).backward()

            if iter % args.gradient_accumulation_steps == 0 or iter == len(dl_train):
                scaler_g.step(net_gene.optimizer)
                scaler_g.update()

            # Save training snapshot for visual tracking
            for _id, _i, _t, _b, _m in zip(img_ids, img_mixs, norm11to01(img_highs), norm11to01(generate_b), masks):
                if saveimage_train in _id:
                    save_image(torch.cat([_i, _t, _b, _m], dim=2), os.path.join(args.ckpt_folder, saveimage_train, f"{str(epoch).zfill(4)}.jpg"))

            loss_gene_fwd_epochs += loss_gene_fwd.item()
            loss_gene_adv_epochs += loss_gene_adv.item()

        loss_gene_fwd_epochs /= len(dl_train)
        loss_gene_adv_epochs /= len(dl_train)
        loss_gene_epochs = abs(loss_gene_adv_epochs) + abs(loss_gene_fwd_epochs)

        # Save model checkpoints
        if loss_gene_best > loss_gene_epochs:
            best_epoch = epoch
            loss_gene_best = loss_gene_epochs
            net_gene.save_networks(bestmodel_cpt, epoch, best_epoch, loss_gene_epochs, loss_gene_best, net_disc, loss_disc.item())

        print('\r epoch: [%3d/%3d] LossG: %.3f LossD: %.3f %s' % (
            epoch, args.num_training_epochs, loss_gene_epochs, loss_disc.item(), "/BEST/" if best_epoch == epoch else ""
        ), end='')

        if epoch % args.ckpt_by_epochs == 0:
            net_gene.save_networks(currentmodel_cpt, epoch, best_epoch, loss_gene_epochs, loss_gene_best, net_disc, loss_disc.item())
        if epoch % args.save_by_epochs == 0:
            shutil.copy2(currentmodel_cpt, os.path.join(args.train_model_folder, f"{str(epoch).zfill(4)}.pth"))

        # Validation Step
        if epoch % args.eval_by_epochs == 0 or best_epoch == epoch:
            net_gene.set_eval()
            net_disc.set_eval()
            
            with torch.no_grad():
                eval_ssim, eval_psnr, eval_lpips, eval_dists = [], [], [], []
                for iter, batch in enumerate(dl_valid):
                    img_id, org_mix, org_he, org_le, img_size = batch
                    org_mix = org_mix.to(args.device)
                    org_he = org_he.to(args.device)
                    
                    gen_he, eval_v = net_gene.get_eval_wo_misaligned(net_gene.gene_a2b, org_mix, org_he, None)
                    eval_ssim.append(eval_v['ssim'])
                    eval_psnr.append(eval_v['psnr'])
                    eval_lpips.append(eval_v['lpips'])
                    eval_dists.append(eval_v['dists'])

                print(f" | VAL | SSIM: {mean(eval_ssim):.3f} PSNR: {mean(eval_psnr):.3f} LPIPS: {mean(eval_lpips):.3f} DISTS: {mean(eval_dists):.3f}")
                with open(loss_txt, 'a') as f:
                    print(f"{epoch},{loss_gene_epochs},{loss_disc.item()},{mean(eval_ssim):.3f},{mean(eval_psnr):.3f},{mean(eval_lpips):.3f},{mean(eval_dists):.3f}", file=f)

        # Check for early stopping
        with open(args.num_training_epochs_txt[0]) as f:
            num_training_epochs = int(f.readlines()[args.num_training_epochs_txt[1]])
            if epoch >= num_training_epochs:
                net_gene.save_networks(currentmodel_cpt, epoch, best_epoch, loss_gene_epochs, loss_gene_best, net_disc, loss_disc.item())
                print(f"Training completed successfully! Saved latest.pth.")
                break

if __name__ == "__main__":
    # Load default configs
    args = parse_args_options(is_train=True)
    # Add non-interactive arguments
    parser = argparse.ArgumentParser(parents=[parse_args_options(is_train=True, input_args=[])], conflict_handler='resolve')
    parser.add_argument("--resume", action="store_true", help="Resume training from latest.pth checkpoint")
    args = parser.parse_args()
    
    main(args)
