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

from datetime import datetime
import os
import numpy as np
from torchmeta.modules import MetaModule

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
    
def inner_loop_update(model, loss, current_step, params=None, inner_lr=0.5, first_order=False):
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)

    updated_params = OrderedDict()

    if isinstance(inner_lr, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            # validation
            if current_step >= len(inner_lr[name]):
                updated_params[name] = param - inner_lr[name][0] * grad
            
            # Training
            else:
                updated_params[name] = param - inner_lr[name][current_step] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - inner_lr * grad

    return updated_params
    
def inner_loop(model, bound, num_samples, raybatch_size, inner_steps, inner_lr, train_data, per_step_loss_importance_vectors=None, first_order=True):
    pixels, rays_o, rays_d, num_rays = train_data['support']
    target_pixels, target_rays_o, target_rays_d, target_num_rays = train_data['target']
    
    params = None
    total_losses = []
    for step in range(inner_steps):
        loss = compute_loss(model, num_rays, raybatch_size, rays_o, rays_d, pixels, num_samples, bound, params)
        model.zero_grad()
        # params = GUP(model, loss, params=params, step_size=inner_lr, first_order=first_order)
        params = inner_loop_update(model, loss, current_step=step, params=params, 
                                    inner_lr=inner_lr, first_order=first_order)
        
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
                    
def report_result(model, test_imgs, test_poses, hwf, bound, raybatch_size, num_samples, tto_showImages, params=None):
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
    
def validate_MAML(model, tto_imgs, tto_poses, test_imgs, test_poses, hwf, bound, inner_lr, args):
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
        loss = compute_loss(model, num_rays, args.tto_batchsize, rays_o, rays_d, pixels, args.num_samples, bound, params)
        
        # indices = torch.randint(num_rays, size=[args.tto_batchsize])
        # raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        # pixelbatch = pixels[indices] 
        # t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
        #                             args.num_samples, perturb=True)
        
        # rgbs, sigmas = model(xyz, params=params)
        # colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        # loss = F.mse_loss(colors, pixelbatch)

        model.zero_grad()
        # OOM when tto_step>100 as MAML need to backprop to all those step
        # https://github.com/tristandeleu/pytorch-maml/issues/21
        # always use first order when validating
        
        # params = GUP(model, loss, params=params, step_size=inner_lr, first_order=True)
        params = inner_loop_update(model, loss, current_step=step, params=params, 
                                    inner_lr=inner_lr, first_order=True)
        
    # use the param from TTO on test imgs
    return report_result(model, test_imgs, test_poses, hwf, bound, 
                args.test_batchsize, args.num_samples, args.tto_showImages, params)    

def to_int(float_tensor, clamp_min=0, clamp_max=32):
    return int(torch.clamp(float_tensor, clamp_min, clamp_max).item())

# python shapenet_train.py --config configs/shapenet/chairs.json
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
    parser.add_argument('--learn_inner_lr', action='store_true',
                        help='the inner lr is a learnable (meta-trained) additional argument')
    # Meta-SGD
    parser.add_argument('--per_param_inner_lr', action='store_true',
                        help='the inner lr parameter is different for each parameter of the model. Has no impact unless `learn_inner_lr=True')
    # MAML++
    parser.add_argument('--use_scheduler', action='store_true',
                        help='use scheduler to adjust outer loop lr')   
    parser.add_argument('--checkpoint_path', type=str,
                        help='path to saved checkpoint')   
    parser.add_argument('--make_checkpoint_dir', action='store_true',
                        help='make a directory in checkpoint_path with name as current time')
    parser.add_argument('--plot_lr', action='store_true',
                        help='plot inner learning rates')      
    parser.add_argument('--load_meta_init', type=str, default="random",
                        help='start meta training with an initialization')  
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
        
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load train & val dataset
    # ========================
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
    # ========================
    
    meta_model = build_nerf(args)
    meta_model.to(device)
    
    if args.load_meta_init != 'random':
        checkpoint = torch.load(args.load_meta_init, map_location=device)
        meta_state = checkpoint['meta_model_state_dict']
        meta_model.load_state_dict(meta_state)
        print(f"load meta_model_state_dict from {args.load_meta_init}")
    
    meta_optim = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
    
    # learnable inner_lr & per_param_inner_lr
    # ============================
    inner_steps = torch.tensor(args.inner_steps, 
        dtype=torch.float, 
        device=device,
        requires_grad=args.learn_inner_step)
    # print(to_int(inner_steps, args.inner_steps_min, args.inner_steps_max))
    log_inner_steps = [inner_steps.item()]
    if args.learn_inner_step:
        meta_optim.add_param_group({'params': [inner_steps]})
    
    inner_lr = args.inner_lr
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=meta_optim, T_max=args.max_iters, eta_min=args.scheduler_min_lr)
                                                          
    # log the first layer's per step learning rate
    inner_per_step_lr = [[] for _ in range(args.inner_steps_max if args.learn_inner_step else args.inner_steps)]
    
    for (name, param) in meta_model.meta_named_parameters():
        first_layer_name = name
        break
    
    if args.per_param_inner_lr:
        # for logging the inner lr
        inner_lrs = OrderedDict((name, [inner_lr]) for (name, param)
            in meta_model.meta_named_parameters())
        
        # for each step and each layer, create a learnable inner lr (LSLR)
        init_inner_lr = torch.ones(args.inner_steps_max if args.learn_inner_step else args.inner_steps) * inner_lr
        init_inner_lr.to(device)
        # inner_lr = OrderedDict()
        # for (name, param) in meta_model.meta_named_parameters():
        #     inner_lr[name] = init_inner_lr.clone().detach().requires_grad_(args.learn_inner_lr).to(device)
            
        inner_lr = OrderedDict((name,
            torch.tensor(init_inner_lr, 
            dtype=param.dtype, 
            device=device,
            requires_grad=args.learn_inner_lr)) 
            for (name, param) in meta_model.meta_named_parameters())
        
    else:
        inner_lr = torch.tensor(inner_lr, dtype=torch.float32,
            device=device, requires_grad=args.learn_inner_lr)        
        
    if args.learn_inner_lr:
        if args.per_param_inner_lr:
            meta_optim.add_param_group({'params': inner_lr.values()})
        else:
            meta_optim.add_param_group({'params': [inner_lr]})
                
        # meta_optim.add_param_group({'params': inner_lr.values()
        #     if args.per_param_inner_lr else [inner_lr]})
    # ============================

    if args.resume_step != 0:
        weight_path = f"{args.checkpoint_path}/step{args.resume_step}.pth"
        checkpoint = torch.load(weight_path, map_location=device)
        meta_state = checkpoint['meta_model_state_dict']
        meta_model.load_state_dict(meta_state)
        inner_lr = checkpoint['inner_lr']
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
    
            # for each scene, run multiple stages
            for _ in range(args.MAML_stage):
                per_step_loss_importance_vectors = None
                if args.use_MSL and step < args.MSL_iterations:
                    per_step_loss_importance_vectors = get_per_step_loss_importance_vector(
                                to_int(inner_steps, args.inner_steps_min, args.inner_steps_max), args.MSL_iterations, step, device)
                
                
                # https://github.com/tristandeleu/pytorch-meta/blob/master/examples/maml/train.py
                outer_loss = torch.tensor(0.).to(device)
                batch_size = args.MAML_batch
                train_data = prepare_MAML_data(imgs, poses, batch_size, hwf)
                                    
                # In MAML, the losses of a batch tasks were used to update meta parameter 
                # but the batch of tasks in NeRF does not makes too much sense
                # should it be a batch of scenes? or a batch of pixels in a single scene
                for i in range(batch_size):
                    # update parameter with the inner loop loss
                    loss = inner_loop(meta_model, bound, args.num_samples,
                        args.train_batchsize, to_int(inner_steps, args.inner_steps_min, args.inner_steps_max), inner_lr, train_data,
                        per_step_loss_importance_vectors=per_step_loss_importance_vectors,
                        first_order= not (args.use_second_order and step>args.first_order_to_second_order_iteration))
                        
                    outer_loss += loss
                    
                pbar.set_postfix({
                    'inner_lr': inner_lr['net.1.weight'][0].item() if args.per_param_inner_lr else inner_lr.item(), 
                    "outer_lr" : scheduler.get_last_lr()[0], 
                    'Train loss': loss.item()
                })
                meta_optim.zero_grad()
                outer_loss.div_(batch_size)
                outer_loss.backward()
            
                meta_optim.step()
                # after step, optimizer will update inner per step learning rate and inner step
                # it makes more sense to use the same scene to get a new loss for the updated params
                train_psnrs.append((step, -10*torch.log10(outer_loss).detach().cpu().item()))
        
            if step % args.val_freq == 0 and step != args.resume_step:
                # show one of the validation result
                test_psnrs = []
                for imgs, poses, hwf, bound in tqdm(val_loader, desc = 'Validating'):
                    imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
                    imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
            
                    tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
                    tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)
                    
                    scene_psnr = validate_MAML(meta_model, tto_imgs, tto_poses, test_imgs, test_poses, hwf, bound, inner_lr, args)            
                                                
                    test_psnrs.append(scene_psnr)
            
                val_psnr = torch.stack(test_psnrs).mean()
                
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
                    
                    # ===================
                    plt.subplots()
                    plt.ylabel("per step learning rate")
                    plt.xlabel("iterations")
                    plt.title(f"per layers per steps inner learning rate for layer {first_layer_name}")
                    for (idx, lrs) in enumerate(inner_per_step_lr):
                        plt.plot(lrs, label=f"inner step {idx}")
                    
                    plt.legend()
                    plt.savefig(f'{args.checkpoint_path}/{step}_per_steps_lr.png')
                    plt.show()
                    
                    # ===================
                    plt.subplots()
                    plt.ylabel("inner steps")
                    plt.xlabel("iterations")
                    plt.title(f"Changes of learned inner steps")
                    plt.plot(log_inner_steps)
                    
                    plt.legend()
                    plt.savefig(f'{args.checkpoint_path}/{step}_inner_steps.png')
                    plt.show()
                    
                
                with open(f'{args.checkpoint_path}/psnr.txt', 'w') as f:
                    psnr = {
                        'train': train_psnrs,
                        'val' : val_psnrs,
                        'best_val': sorted(val_psnrs,key=lambda x: x[1], reverse=True)[0]
                    }
                    f.write(json.dumps(psnr))
          
            if step % args.checkpoint_freq == 0 and step != args.resume_step:
                path = f"{args.checkpoint_path}/step{step}.pth"
                
                torch.save({
                    'epoch': step,
                    'meta_model_state_dict': meta_model.state_dict(),
                    'meta_optim_state_dict': meta_optim.state_dict(),
                    'inner_lr': inner_lr,
                    }, path)
                        
                print(f"step{step} model save to {path}")
            
            if args.use_scheduler:
                scheduler.step()
                
                if args.plot_lr:
                    for (name, param) in meta_model.meta_named_parameters():
                        inner_lrs[name].append(inner_lr[name][0].item())
                    
                    for idx, lr_list in enumerate(inner_per_step_lr):
                        lr_list.append(inner_lr[first_layer_name][idx].item())
                    
                    log_inner_steps.append(inner_steps.item())
                
            step += 1
            pbar.update(1)
            
            if step > args.max_iters:
              break
        
    pbar.close()

if __name__ == '__main__':
    main()