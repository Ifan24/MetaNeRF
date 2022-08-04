from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from utils.shape_video import create_360_video
from models.rendering import get_rays_shapenet, sample_points, volume_render
try:
  import google.colab
  from tqdm.notebook import tqdm as tqdm
except:
  from tqdm import tqdm as tqdm
  
import matplotlib.pyplot as plt
from shapenet_train import validate_MAML, compute_loss, report_result
from torchmeta.utils.gradient_based import gradient_update_parameters as GUP


def train_val_scene(args, model, tto_imgs, tto_poses, test_imgs, test_poses, hwf, bound, inner_lr=0.5):
    """
    train and val the model on available views
    """
    train_val_freq = args.train_val_freq
    # 25x128x128
    pixels = tto_imgs.reshape(-1, 3)
    rays_o, rays_d = get_rays_shapenet(hwf, tto_poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    num_rays = rays_d.shape[0]

    if args.per_param_inner_lr:
        params = []
        for (name, param) in model.meta_named_parameters():
            params.append({
                'params': param,
                'lr': inner_lr[name][0].item()
            })
        optim = torch.optim.SGD(params, args.tto_lr)
    else:
        optim = torch.optim.SGD(model.parameters(), args.tto_lr)
    
    val_psnrs = []
    for step in tqdm(range(args.train_val_steps), desc = 'Train & Validate'):
        optim.zero_grad()
        loss = compute_loss(model=model, num_rays=num_rays, raybatch_size=args.tto_batchsize, 
                rays_o=rays_o, rays_d=rays_d, pixels=pixels, 
                num_samples=args.num_samples, bound=bound)
        loss.backward()
        optim.step()
        
        
        if step % train_val_freq == 0 and step != 0:
            with torch.no_grad():
                scene_psnr = report_result(model, test_imgs, test_poses, hwf, bound, 
                                args.test_batchsize, args.num_samples, args.tto_showImages)
            
            # Plot validation PSNR
            val_psnrs.append((step, scene_psnr.item()))
            print(f"step: {step}, val psnr: {scene_psnr:0.3f}")
            plt.plot(*zip(*val_psnrs), label="val_psnr")
            plt.title(f'ShapeNet Reconstruction from {args.tto_views} views')
            plt.xlabel('Iterations')
            plt.ylabel('PSNR')
            plt.legend()
            plt.show()

        if step <= 1000:
            train_val_freq = 100
        elif step > 1000 and step <= 10000:
            train_val_freq = 500
        elif step > 10000 and step <= 50000:
            train_val_freq = 2500
        elif step > 50000 and step <= 100000:
            train_val_freq = 5000
    print(val_psnrs)


def test():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                    help='config file for the shape class (cars, chairs or lamps)')    
    parser.add_argument('--weight-path', type=str, required=True,
                        help='path to the meta-trained weight file')
    parser.add_argument('--one_scene', action='store_true', help="train and validate the model on the first scene of test dataset")
    parser.add_argument('--standard_init', action='store_true', help="train and validate the model without meta learning parameters")
    parser.add_argument('--meta', type=str, default='Reptile', choices=['MAML', 'Reptile'],
                        help='meta algorithm, (MAML, Reptile)')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = build_shapenet(image_set="test", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    if args.max_test_size != 0:
        test_set = Subset(test_set, range(0, args.max_test_size))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = build_nerf(args)
    model.to(device)

    inner_lr = args.tto_lr
    if not args.standard_init:
        checkpoint = torch.load(args.weight_path, map_location=device)
        meta_state = checkpoint['meta_model_state_dict']
        if args.per_param_inner_lr:
            inner_lr = checkpoint['inner_lr']

    savedir = Path(args.savedir)
    savedir.mkdir(exist_ok=True)
    
    if args.one_scene:
        for imgs, poses, hwf, bound in test_loader:
            imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
            imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
            tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
            tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)
            
            if not args.standard_init:
                model.load_state_dict(meta_state)
                
            train_val_scene(args, model, tto_imgs, tto_poses, test_imgs, test_poses, hwf, bound, inner_lr)
            
            return
    
    test_psnrs = []
    idx = 0
    pbar = tqdm(test_loader, desc = 'Testing')
    for imgs, poses, hwf, bound in pbar:
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        model.load_state_dict(meta_state)
        scene_psnr = validate_MAML(model, tto_imgs, tto_poses, test_imgs, test_poses, 
                hwf, bound, inner_lr, args)
        
        if args.create_video:
            create_360_video(args, model, hwf, bound, device, idx+1, savedir)
            print(f"scene {idx+1}, psnr:{scene_psnr:.3f}, video created")
        else:
            print(f"scene {idx+1}, psnr:{scene_psnr:.3f}")
            
        test_psnrs.append(scene_psnr)
        
        pbar.set_postfix(mean_psnr=torch.stack(test_psnrs).mean().item())
        idx += 1
    
    test_psnrs = torch.stack(test_psnrs)
    print("----------------------------------")
    print(f"test dataset mean psnr: {test_psnrs.mean():.3f}")


if __name__ == '__main__':
    test()