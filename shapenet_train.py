import argparse
import json
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from models.rendering import get_rays_shapenet, sample_points, volume_render
try:
  import google.colab
  from tqdm.notebook import tqdm as tqdm
except:
  from tqdm import tqdm as tqdm
  
import matplotlib.pyplot as plt
from torchmeta.utils.gradient_based import gradient_update_parameters as GUP
import matplotlib.pyplot as plt
from collections import OrderedDict
import gc

from datetime import datetime
import os
import numpy as np
from torchmeta.modules import MetaModule

def prepare_MAML_data(imgs, poses, batch_size, hwf):
    '''
        split training images to support set and target set
        size(support set) = train_views - MAML_batch_size
        size(target set) = MAML_batch_size
    '''
    # shuffle the images
    indexes = torch.randperm(imgs.shape[0])
    imgs = imgs[indexes]
    poses = poses[indexes]
    
    target_imgs, target_poses = imgs[:batch_size], poses[:batch_size]
    imgs, poses = imgs[batch_size:], poses[batch_size:] 
    
    pixels = imgs.reshape(-1, 3)
    # 25x128x128
    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    num_rays = rays_d.shape[0]
    
    target_pixels = target_imgs.reshape(-1, 3)
    target_rays_o, target_rays_d = get_rays_shapenet(hwf, target_poses)
    target_rays_o, target_rays_d = target_rays_o.reshape(-1, 3), target_rays_d.reshape(-1, 3)
    target_num_rays = target_rays_d.shape[0]
    
    return {
            'support': [pixels, rays_o, rays_d, num_rays], 
            'target':[target_pixels, target_rays_o, target_rays_d, target_num_rays]
        }
                    
def compute_loss(model, num_rays, raybatch_size, rays_o, rays_d, pixels, num_samples, bound, params=None):
    indices = torch.randint(num_rays, size=[raybatch_size])
    raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
    pixelbatch = pixels[indices] 
    t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                num_samples, perturb=True)
                                
    if isinstance(model, MetaModule):
        rgbs, sigmas = model(xyz, params=params)
    else:
       rgbs, sigmas = model(xyz)

    colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
    loss = F.mse_loss(colors, pixelbatch)
    return loss
    
def MAML_inner_loop(model, bound, num_samples, raybatch_size, inner_steps, alpha, train_data, per_step_loss_importance_vectors=None, first_order=True):
    
        
    pixels, rays_o, rays_d, num_rays = train_data['support']
    target_pixels, target_rays_o, target_rays_d, target_num_rays = train_data['target']
    
    params = None
    total_losses = []
    for step in range(inner_steps):
        loss = compute_loss(model, num_rays, raybatch_size, rays_o, rays_d, pixels, num_samples, bound, params)
        model.zero_grad()
        params = GUP(model, loss, params=params, step_size=alpha, first_order=first_order)
        
        if per_step_loss_importance_vectors is not None:
            loss = compute_loss(model, target_num_rays, raybatch_size, target_rays_o, target_rays_d, target_pixels, num_samples, bound, params)
            total_losses.append(per_step_loss_importance_vectors[step] * loss)
        
    if per_step_loss_importance_vectors is None:
        # use the param from previous inner_steps on val views to get a outer loss
        final_loss = compute_loss(model, target_num_rays, raybatch_size, target_rays_o, target_rays_d, target_pixels, num_samples, bound, params)
    else:
        # use MSL
        final_loss = torch.sum(torch.stack(total_losses))
             
    return final_loss
                    
def inner_loop(model, optim, imgs, poses, hwf, bound, num_samples, raybatch_size, inner_steps):
    """
    train the inner model for a specified number of iterations
    """
    pixels = imgs.reshape(-1, 3)

    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    for step in range(inner_steps):
        indices = torch.randint(num_rays, size=[raybatch_size])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()
    
def report_result(model, imgs, poses, hwf, bound, num_samples, raybatch_size, tto_showImages):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)
    view_psnrs = []
    plt.figure(figsize=(15, 6))
    count = 0
    for img, rays_o, rays_d in zip(imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    num_samples, perturb=False)
        
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, raybatch_size):
                rgbs_batch, sigmas_batch = model(xyz[i:i+raybatch_size])
                color_batch = volume_render(rgbs_batch, sigmas_batch, 
                                            t_vals[i:i+raybatch_size],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.clip(torch.cat(synth, dim=0).reshape_as(img), min=0, max=1)
            error = F.mse_loss(img, synth)
            psnr = -10*torch.log10(error)
            view_psnrs.append(psnr)
            
            if count < tto_showImages:
                plt.subplot(2, 5, count+1)
                plt.imshow(img.cpu())
                plt.title('Target')
                plt.subplot(2,5,count+6)
                plt.imshow(synth.cpu())
                plt.title(f'synth psnr:{psnr:0.2f}')
            count += 1
            
    plt.show()
    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr


def render_and_psnr(model, test_imgs, test_poses, hwf, bound, raybatch_size, num_samples, tto_showImages, params=None):
    """
        render images in test_poses and compute the PSNR against the test_imgs
        using the params as the model weight if provided, if not then use the model weight
        :param model: meta nerf model
        :param test_imgs: ground truth images 
        :param test_poses: ground truth images poses
        :param hwf: height, width, focal
        :param bound: bound of the scene 
        
        :return: mean PSNR of the scene.
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, test_poses)
    view_psnrs = []
    plt.figure(figsize=(15, 6))
    count = 0
    for img, rays_o, rays_d in zip(test_imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    num_samples, perturb=False)
        
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, raybatch_size):
                if isinstance(model, MetaModule):
                    rgbs_batch, sigmas_batch = model(xyz[i:i+raybatch_size], params=params)
                else:
                    rgbs_batch, sigmas_batch = model(xyz[i:i+raybatch_size])
                    
                color_batch = volume_render(rgbs_batch, sigmas_batch, 
                                            t_vals[i:i+raybatch_size],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.clip(torch.cat(synth, dim=0).reshape_as(img), min=0, max=1)
            error = F.mse_loss(img, synth)
            psnr = -10*torch.log10(error)
            view_psnrs.append(psnr)
            
            if count < tto_showImages:
                plt.subplot(2, 5, count+1)
                plt.imshow(img.cpu())
                plt.title('Target')
                plt.subplot(2,5,count+6)
                plt.imshow(synth.cpu())
                plt.title(f'synth psnr:{psnr:0.2f}')
            count += 1
            
    plt.show()
    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr  
    
def validate_MAML(model, tto_imgs, tto_poses, test_imgs, test_poses, hwf, bound, step_size, args):
    '''
        train and report the result of model
    '''
    # prepare data
    pixels = tto_imgs.reshape(-1, 3)
    rays_o, rays_d = get_rays_shapenet(hwf, tto_poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    num_rays = rays_d.shape[0]
    # TTO training
    params = None
    
    for step in range(args.tto_steps):
        indices = torch.randint(num_rays, size=[args.tto_batchsize])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    args.num_samples, perturb=True)
        
        rgbs, sigmas = model(xyz, params=params)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)

        model.zero_grad()
        # OOM when tto_step>100 as MAML need to backprop to all those step
        # https://github.com/tristandeleu/pytorch-maml/issues/21
        # always use first order when validating
        params = GUP(model, loss, params=params, step_size=step_size, first_order=True)
        
        
    # use the param from TTO on test imgs
    return render_and_psnr(model, test_imgs, test_poses, hwf, bound, 
                args.test_batchsize, args.num_samples, args.tto_showImages, params)    

def val_meta(args, model, val_loader, device, step_size=None):
    """
    validate the meta trained model for few-shot view synthesis
    """
    meta_trained_state = model.state_dict()
    val_model = copy.deepcopy(model)
    # show one of the validation result
    val_psnrs = []
    for imgs, poses, hwf, bound in tqdm(val_loader, desc = 'Validating'):
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        val_model.load_state_dict(meta_trained_state)
        
        if args.per_param_step_size:
            params = []
            for (name, param) in val_model.meta_named_parameters():
                params.append({
                    'params': param,
                    'lr': step_size[name].item()
                })
            val_optim = torch.optim.SGD(params, args.tto_lr)
        else:
            val_optim = torch.optim.SGD(val_model.parameters(), args.tto_lr)

        inner_loop(val_model, val_optim, tto_imgs, tto_poses, hwf,
                    bound, args.num_samples, args.tto_batchsize, args.tto_steps)      
        scene_psnr = report_result(val_model, test_imgs, test_poses, hwf, bound, 
                                    args.num_samples, args.test_batchsize, args.tto_showImages)
        val_psnrs.append(scene_psnr)

    val_psnr = torch.stack(val_psnrs).mean()
    return val_psnr


def get_per_step_loss_importance_vector(inner_steps, MSL_iterations, current_iteration, device):
    """
    Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
    loss towards the optimization loss.
    :param inner_steps: number of inner steps
    :param MSL_iterations: number of iterations we use MSL
    :return: A tensor to be used to compute the weighted average of the loss, useful for
    the MSL (Multi Step Loss) mechanism.
    """
    loss_weights = np.ones(shape=(inner_steps)) * (1.0 / inner_steps)
    decay_rate = 1.0 / inner_steps / MSL_iterations
    min_value_for_non_final_losses = 0.03 / inner_steps
    for i in range(len(loss_weights) - 1):
        curr_value = np.maximum(loss_weights[i] - (current_iteration * decay_rate), min_value_for_non_final_losses)
        loss_weights[i] = curr_value

    curr_value = np.minimum(
        loss_weights[-1] + (current_iteration * (inner_steps - 1) * decay_rate),
        1.0 - ((inner_steps - 1) * min_value_for_non_final_losses))
    loss_weights[-1] = curr_value
    loss_weights = torch.Tensor(loss_weights).to(device=device)
    return loss_weights
        
def main():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the shape class (cars, chairs or lamps)')
    parser.add_argument('--resume_step', type=int, default=0,
                        help='resume training from step')
    parser.add_argument('--meta', type=str, default='Reptile', choices=['MAML', 'Reptile'],
                        help='meta algorithm, (MAML, Reptile)')
    parser.add_argument('--MAML_batch', type=int, default=3,
                        help='number of batch of task for MAML')
    # Meta-SGD
    parser.add_argument('--learn_step_size', action='store_true',
                        help='the step size is a learnable (meta-trained) additional argument')
    # Meta-SGD
    parser.add_argument('--per_param_step_size', action='store_true',
                        help='the step size parameter is different for each parameter of the model. Has no impact unless `learn_step_size=True')
    # MAML++
    parser.add_argument('--use_scheduler', action='store_true',
                        help='use scheduler to adjust outer loop lr')   
    parser.add_argument('--checkpoint_path', type=str,
                        help='path to saved checkpoint')   
    parser.add_argument('--make_checkpoint_dir', action='store_true',
                        help='make a directory in checkpoint_path with name as current time')
    parser.add_argument('--plot_lr', action='store_true',
                        help='plot inner learning rates')                 
    args = parser.parse_args()
    
    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value
    
    print(vars(args))
    
    if args.make_checkpoint_dir:
        # dd/mm/YY H:M:S
        now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        checkpoint_path = f"{args.checkpoint_path}/{now}"
        os.makedirs(checkpoint_path, exist_ok=True)
        print(f"make directory {checkpoint_path}")
        args.checkpoint_path = checkpoint_path
        
        with open(f'{args.checkpoint_path}/config.json', 'w') as fp:
            json.dump(vars(args), fp)
        
        
    use_reptile = args.meta == 'Reptile'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load train & val dataset
    train_set = build_shapenet(image_set="train", dataset_root=args.dataset_root,
                            splits_path=args.splits_path, num_views=args.train_views)
    if args.max_train_size != 0:
        train_set = Subset(train_set, range(0, args.max_train_size))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    val_set = build_shapenet(image_set="val", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    if args.max_val_size != 0:
        val_set = Subset(val_set, range(0, args.max_val_size))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    meta_model = build_nerf(args)
    meta_model.to(device)
    
    meta_optim = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
    
    # learn_step_size & per_param_step_size
    step_size = args.inner_lr
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=meta_optim, T_max=args.max_iters, eta_min=args.scheduler_min_lr)
                                                              
    if args.per_param_step_size:
        inner_lrs = OrderedDict((name, [step_size]) for (name, param)
            in meta_model.meta_named_parameters())
        step_size = OrderedDict((name, torch.tensor(step_size,
            dtype=param.dtype, device=device,
            requires_grad=args.learn_step_size)) for (name, param)
            in meta_model.meta_named_parameters())
    else:
        step_size = torch.tensor(step_size, dtype=torch.float32,
            device=device, requires_grad=args.learn_step_size)
            
    if args.learn_step_size:
        meta_optim.add_param_group({'params': step_size.values()
            if args.per_param_step_size else [step_size]})
         
    if args.resume_step != 0:
        weight_path = f"{args.checkpoint_path}/step{args.resume_step}.pth"
        checkpoint = torch.load(weight_path, map_location=device)
        meta_state = checkpoint['meta_model_state_dict']
        meta_model.load_state_dict(meta_state)
        step_size = checkpoint['step_size']
        meta_optim.load_state_dict(checkpoint['meta_optim_state_dict'])
        
        print(f"load meta_model_state_dict from {weight_path}")
        
    step = args.resume_step
    pbar = tqdm(total=args.max_iters, desc = 'Training')
    pbar.update(args.resume_step)
    val_psnrs = []
    train_psnrs = []
    while step < args.max_iters:
        for imgs, poses, hwf, bound in train_loader:                    
            # imgs = [1, train_views(25), H(128), W(128), C(3)]
            imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
            imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
    
            meta_optim.zero_grad()
            
            per_step_loss_importance_vectors = None
            if args.use_MSL and step < args.MSL_iterations:
                per_step_loss_importance_vectors = get_per_step_loss_importance_vector(args.inner_steps, args.MSL_iterations, step, device)
            
            if use_reptile:
                inner_model = copy.deepcopy(meta_model)
                if args.per_param_step_size:
                    # TODO: not working 
                    params = []
                    for (name, param) in inner_model.meta_named_parameters():
                        params.append({
                            'params': param,
                            'lr': step_size[name].item()
                        })
                    inner_optim = torch.optim.SGD(params, args.inner_lr)
                else:
                    inner_optim = torch.optim.SGD(inner_model.parameters(), args.inner_lr)
                    
                inner_loop(inner_model, inner_optim, imgs, poses,
                            hwf, bound, args.num_samples,
                            args.train_batchsize, args.inner_steps)

                # Reptile
                with torch.no_grad():
                    for meta_param, inner_param in zip(meta_model.parameters(), inner_model.parameters()):
                        meta_param.grad = meta_param - inner_param
                
                if args.per_param_step_size:
                    pbar.set_postfix({
                        'inner_lr': step_size['net.1.weight'].item() if args.per_param_step_size else step_size.item(), 
                        "outer_lr" : scheduler.get_last_lr()[0], 
                    })
            # python shapenet_train.py --config configs/shapenet/chairs.json
            # MAML
            # https://github.com/tristandeleu/pytorch-meta/blob/master/examples/maml/train.py
            else:                            
                outer_loss = torch.tensor(0.).to(device)
                batch_size = args.MAML_batch
                train_data = prepare_MAML_data(imgs, poses, batch_size, hwf)
                                    
                # In MAML, the losses of a batch tasks were used to update meta parameter 
                # but the batch of tasks in NeRF does not makes too much sense
                # should it be a batch of scenes? or a batch of pixels in a single scene
                for i in range(batch_size):
                    # update parameter with the inner loop loss
                    loss = MAML_inner_loop(meta_model, bound, args.num_samples,
                        args.train_batchsize, args.inner_steps, step_size, train_data,
                        per_step_loss_importance_vectors,
                        first_order= not (args.use_second_order and step>args.first_order_to_second_order_iteration))
                        
                    pbar.set_postfix({
                        'inner_lr': step_size['net.1.weight'].item() if args.per_param_step_size else step_size.item(), 
                        "outer_lr" : scheduler.get_last_lr()[0], 
                        'Train loss': loss.item()
                    })
        
                    # if per_step_loss_importance_vectors is not None:
                    #     outer_loss += per_step_loss_importance_vectors[i] * loss
                    # else:
                    outer_loss += loss
                    
                meta_optim.zero_grad()
                outer_loss.div_(batch_size)
                outer_loss.backward()
        
            meta_optim.step()
        
            if step % args.val_freq == 0 and step != args.resume_step:
                if args.meta == 'MAML' and (args.per_param_step_size or args.learn_step_size):
                    train_psnrs.append((step, -10*torch.log10(outer_loss).detach().cpu().item()))
                    # show one of the validation result
                    test_psnrs = []
                    for imgs, poses, hwf, bound in tqdm(val_loader, desc = 'Validating'):
                        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
                        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
                
                        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
                        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)
                        
                        scene_psnr = validate_MAML(meta_model, tto_imgs, tto_poses, test_imgs, test_poses, hwf, bound, step_size, args)            
                                                    
                        test_psnrs.append(scene_psnr)
                
                    val_psnr = torch.stack(test_psnrs).mean()
                else:
                    val_psnr = val_meta(args, meta_model, val_loader, device, step_size)
                
                print(f"step: {step}, val psnr: {val_psnr:0.3f}")
                val_psnrs.append((step, val_psnr.cpu().item()))
                
                plt.subplots()
                plt.plot(*zip(*val_psnrs), label="Meta learning validation PSNR")
                plt.plot(*zip(*train_psnrs), label="Meta learning Training PSNR")
                plt.title('ShapeNet Meta learning Training PSNR')
                plt.xlabel('Iterations')
                plt.ylabel('PSNR')
                plt.legend()
                plt.savefig(f'{args.checkpoint_path}/{step}.png')
                plt.show()
                print(val_psnrs)
                
                if args.plot_lr:
                    plt.subplots()
                    plt.ylabel("inner learning rate")
                    plt.xlabel("iterations")
                    plt.title("per layers inner learning rate")
                    for (name, lrs) in inner_lrs.items():
                        plt.plot(lrs, label=name)
                    
                    plt.legend()
                    plt.savefig(f'{args.checkpoint_path}/{step}_lr.png')
                    plt.show()
                
                with open(f'{args.checkpoint_path}/psnr.txt', 'w') as f:
                    psnr = {
                        'train': train_psnrs,
                        'val' : val_psnrs
                    }
                    f.write(json.dumps(psnr))
          
            if step % args.checkpoint_freq == 0 and step != args.resume_step:
                path = f"{args.checkpoint_path}/step{step}.pth"
                
                if args.learn_step_size or args.per_param_step_size:
                    torch.save({
                        'epoch': step,
                        'meta_model_state_dict': meta_model.state_dict(),
                        'meta_optim_state_dict': meta_optim.state_dict(),
                        'step_size': step_size,
                        }, path)
                else:
                    torch.save({
                        'epoch': step,
                        'meta_model_state_dict': meta_model.state_dict(),
                        'meta_optim_state_dict': meta_optim.state_dict(),
                        }, path)
                        
                print(f"step{step} model save to {path}")
            
            if args.use_scheduler:
                scheduler.step()
                
                if args.plot_lr:
                    for (name, param) in meta_model.meta_named_parameters():
                        inner_lrs[name].append(step_size[name].item())
                
            step += 1
            pbar.update(1)
            
            if step > args.max_iters:
              break
        
    pbar.close()
    print(val_psnrs)
    

if __name__ == '__main__':
    main()