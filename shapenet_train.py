import argparse
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from models.rendering import get_rays_shapenet, sample_points, volume_render
from utils.utils import make_dir
try:
  import google.colab
  from tqdm.notebook import tqdm as tqdm
except:
  from tqdm import tqdm as tqdm
  
import matplotlib.pyplot as plt
from torchmeta.utils.gradient_based import gradient_update_parameters as GUP
import matplotlib.pyplot as plt
from collections import OrderedDict

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

def shuffle_imgs(imgs, poses):
    """shuffle the images and poses, assume there are more than one images

    Returns:
        choose one images as target image and return the rest
    """
        
    # shuffle the images
    indexes = torch.randperm(imgs.shape[0])
    imgs = imgs[indexes]
    poses = poses[indexes]
    
    target_imgs = imgs[0]
    target_poses = poses[0]
    
    imgs = imgs[1:]
    poses = poses[1:]
    return imgs, poses, target_imgs, target_poses

def prepare_MAML_data(imgs, poses, batch_size, hwf, device):
    '''
        divided data into batches with len batch_size
        and in each chunk split training images to support set and target set
        size(support set) = train_pixels - raybatch_size
        size(target set) = raybatch_size
    '''
    # takes 1 view as target set
    # assume the MAML batch < 128 and training view > 1
    
    # shuffle the images
    # indexes = torch.randperm(imgs.shape[0])
    # imgs = imgs[indexes]
    # poses = poses[indexes]
    
    # target_imgs = imgs[0]
    # target_poses = poses[0]
    
    # imgs = imgs[1:]
    # poses = poses[1:]
    
    imgs, poses, target_imgs, target_poses = shuffle_imgs(imgs, poses)
    
    def shuffle_pixels(imgs, poses):
        pixels = imgs.reshape(-1, 3)
        rays_o, rays_d = get_rays_shapenet(hwf, poses)
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        indexes = torch.randperm(pixels.shape[0])
        return pixels[indexes].chunk(batch_size), rays_o[indexes].chunk(batch_size), rays_d[indexes].chunk(batch_size)
        
    pixels, rays_o, rays_d = shuffle_pixels(imgs, poses)
    target_pixels, target_rays_o, target_rays_d = shuffle_pixels(target_imgs, target_poses)
    batches = []
    for i in range(batch_size):
        train_data = {
                'support': [pixels[i].to(device), rays_o[i].to(device), rays_d[i].to(device), pixels[i].shape[0]],
                'target':[target_pixels[i].to(device), target_rays_o[i].to(device), target_rays_d[i].to(device), target_pixels[i].shape[0]]
        }
        batches.append(train_data)
        
    return batches
    
                    
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
    
def inner_loop_update(model, loss, current_step, params=None, inner_lr=0.5, init_lr=0.5, first_order=False):
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
            # if current_step >= len(inner_lr[name]):
            #     updated_params[name] = param - init_lr * grad
            
            # # Training
            # else:
            #     updated_params[name] = param - inner_lr[name][current_step] * grad
            updated_params[name] = param - inner_lr[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - inner_lr * grad

    return updated_params
    
def inner_loop(model, bound, num_samples, raybatch_size, inner_steps, inner_lr, train_data, per_step_loss_importance_vectors=None, first_order=True, init_lr=0.5):
    pixels, rays_o, rays_d, num_rays = train_data['support']
    target_pixels, target_rays_o, target_rays_d, target_num_rays = train_data['target']
    
    params = None
    total_losses = []
    for step in range(inner_steps):
        loss = compute_loss(model, num_rays, raybatch_size, rays_o, rays_d, pixels, num_samples, bound, params)
        model.zero_grad()
        # params = GUP(model, loss, params=params, step_size=inner_lr, first_order=first_order)
        params = inner_loop_update(model, loss, current_step=step, params=params, 
                                    inner_lr=inner_lr, init_lr=init_lr, first_order=first_order)
        
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
                    
def report_result(model, test_imgs, test_poses, hwf, bound, raybatch_size, num_samples, tto_showImages, params=None, show_img=True):
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
    if show_img:
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
            
            if show_img and count < tto_showImages:
                plt.subplot(2, 5, count+1)
                plt.imshow(img.cpu())
                plt.title('Target')
                plt.subplot(2,5,count+6)
                plt.imshow(synth.cpu())
                plt.title(f'synth psnr:{psnr:0.2f}')
            count += 1
    
    if show_img:
        plt.show()
        
    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr  
    
def validate_MAML(model, tto_imgs, tto_poses, test_imgs, test_poses, hwf, bound, inner_lr, args):
    """train on the tto images and test on the test images, using the inner_lr for training

    Returns:
        float: PSNR of test images
    """

    
    if tto_imgs.shape[0] == 1:
        plt.imshow(tto_imgs[0].cpu())
        plt.title('The only input Image')
        plt.show()
        
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
                                    inner_lr=inner_lr, init_lr=args.tto_lr, first_order=True)
        
    # use the param from TTO on test imgs
    return report_result(model, test_imgs, test_poses, hwf, bound, 
                args.test_batchsize, args.num_samples, args.tto_showImages, params)    

# based on the past N task to update the inner steps
def update_inner_steps(task_difficulty, min_steps=1, max_steps=16, cache_size=100):
    # only keep track of last N task
    
    last_task = task_difficulty[len(task_difficulty)-1]
    current_task = task_difficulty[-cache_size:]
    current_task.sort()
    difficulty_rank = current_task.index(last_task)
    # difficulty_rank is in range of (0, cache_size)
    
    def remap(x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    new_inner_steps = remap(difficulty_rank, 0, cache_size, min_steps, max_steps)
    return int(new_inner_steps), difficulty_rank
     
def inner_loop_Reptile(model, imgs, poses, hwf, bound, num_samples, raybatch_size, inner_steps, inner_lr, init_lr):
    """
    train the inner model for a specified number of iterations
    """
    # if imgs.shape[0] > 1:
    #     imgs, poses, target_imgs, target_poses = shuffle_imgs(imgs, poses)
    # else:
    #     # print("only one image")
    #     # TTO view = 1 or SV meta learning
    #     target_imgs, target_poses = imgs, poses
    
    pixels = imgs.reshape(-1, 3)
    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    num_rays = rays_d.shape[0]
    
    params = None
    for step in range(inner_steps):
        loss = compute_loss(model, num_rays, raybatch_size, rays_o, rays_d, pixels, num_samples, bound, params)
        model.zero_grad()
        params = inner_loop_update(model, loss, current_step=step, params=params, 
                                    inner_lr=inner_lr, init_lr=init_lr, first_order=True)
                               
    model.load_state_dict(params, strict=False)
    
    # pixels = target_imgs.reshape(-1, 3)
    # rays_o, rays_d = get_rays_shapenet(hwf, target_poses)
    # rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    # num_rays = rays_d.shape[0]
    
    # return compute_loss(model, num_rays, raybatch_size, rays_o, rays_d, pixels, num_samples, bound, params)
    
    
# def inner_loop_Reptile(model, optim, imgs, poses, hwf, bound, num_samples, raybatch_size, inner_steps):
#     pixels = imgs.reshape(-1, 3)
#     rays_o, rays_d = get_rays_shapenet(hwf, poses)
#     rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
#     num_rays = rays_d.shape[0]
    
#     for step in range(inner_steps):
#         optim.zero_grad()
#         loss = compute_loss(model, num_rays, raybatch_size, rays_o, rays_d, pixels, num_samples, bound)
#         loss.backward()
#         optim.step()
        
    
def apply_lr(model, inner_lr, init_lr, inner_step):
    """apply inner learning rate for each layer
    """
    params = []
    for (name, param) in model.meta_named_parameters():
        params.append({
            'params': param,
            'lr': inner_lr[name][inner_step].item()
        })
    optim = torch.optim.SGD(params, init_lr)
    return optim
    
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
    parser.add_argument('--per_layer_inner_lr', action='store_true',
                        help='the inner lr parameter is different for each layers of the model. Has no impact unless `learn_inner_lr=True')
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
        args.checkpoint_path = make_dir(args)
        
        
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
    
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=meta_optim, T_max=args.max_iters, eta_min=args.scheduler_min_lr)
        
    inner_steps = args.inner_steps
    
    log_inner_steps = [inner_steps]
    log_difficulty_rank = []
    # if args.learn_inner_step:
    #     meta_optim.add_param_group({'params': [inner_steps]})
    
    # learnable inner_lr & per_layer_inner_lr
    # ============================
    # log the first layer's per step learning rate
    inner_lr = args.inner_lr
    # inner_per_step_lr = [[] for _ in range(args.inner_steps_max if args.learn_inner_step else args.inner_steps)]
    
    for (name, param) in meta_model.meta_named_parameters():
        first_layer_name = name
        break
    
    if args.per_layer_inner_lr:
        # for logging the inner lr
        inner_lrs = OrderedDict((name, [inner_lr]) for (name, param)
            in meta_model.meta_named_parameters())
        
        # for each step and each layer, create a learnable inner lr (LSLR)
        # init_inner_lr = torch.ones(args.inner_steps_max if args.learn_inner_step else args.inner_steps) * inner_lr
        # init_inner_lr.to(device)
        # inner_lr = OrderedDict()
        # for (name, param) in meta_model.meta_named_parameters():
        #     inner_lr[name] = init_inner_lr.clone().detach().requires_grad_(args.learn_inner_lr).to(device)
            
        # inner_lr = OrderedDict((name,
        #     torch.tensor(init_inner_lr, 
        #     dtype=param.dtype, 
        #     device=device,
        #     requires_grad=args.learn_inner_lr)) 
        #     for (name, param) in meta_model.meta_named_parameters())
        
        # meta-SGD per param lr
        inner_lr = OrderedDict()
        for (name, param) in meta_model.meta_named_parameters():
            # inner_lr[name] = args.inner_lr * torch.ones_like(param, requires_grad=True)
            inner_lr[name] = nn.Parameter(args.inner_lr * torch.ones_like(param, requires_grad=True))
        
    else:
        inner_lr = torch.tensor(inner_lr, dtype=torch.float32,
            device=device, requires_grad=args.learn_inner_lr)
        inner_lrs = []
        inner_lrs.append(inner_lr.item())
                        
    # ============================
    val_psnrs = []
    train_psnrs = []
    task_difficulty = []
    
    if args.resume_step != 0:
        weight_path = f"{args.checkpoint_path}/step{args.resume_step}.pth"
        checkpoint = torch.load(weight_path, map_location=device)
        meta_state = checkpoint['meta_model_state_dict']
        meta_model.load_state_dict(meta_state)
        inner_lr = checkpoint['inner_lr']
        meta_optim.load_state_dict(checkpoint['meta_optim_state_dict'])
        
        with open(f'{args.checkpoint_path}/psnr.txt') as f:
            txt = json.load(f)
            train_psnrs = txt['train']
            val_psnrs = txt['val']
            inner_lrs = txt['inner_lrs']
            # psnr = {
            #     'train': train_psnrs,
            #     'val' : val_psnrs,
            #     'best_val': sorted(val_psnrs,key=lambda x: x[1], reverse=True)[0]
            # }
        print(val_psnrs)
        
        print(f"load meta_model_state_dict from {weight_path}")
    
    # print(meta_optim.param_groups)
    
    if args.learn_inner_lr:
        
        if args.per_layer_inner_lr:
            if args.meta == 'MAML':
                meta_optim.add_param_group({'params': inner_lr.values()})
            else:
                lr_optim = torch.optim.Adam(inner_lr.values(), lr=args.meta_lr)
        else:
            if args.meta == 'MAML':
                meta_optim.add_param_group({'params': [inner_lr]})
            else:
                lr_optim = torch.optim.Adam([inner_lr], lr=args.meta_lr)
                
    step = args.resume_step
    pbar = tqdm(total=args.max_iters, desc = 'Training')
    pbar.update(args.resume_step)

    while step < args.max_iters:
        for imgs, poses, hwf, bound in train_loader:                    
            # imgs = [1, train_views(25), H(128), W(128), C(3)]
            imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
            imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
            # TODO: for each scene, run multiple stages (not sure how it does)
            if args.meta == 'MAML':
                # after step, optimizer will update inner per step learning rate and inner step
                # it makes more sense to use the same scene to get a new loss for the updated hyper params
                for _ in range(args.MAML_stage):
                    per_step_loss_importance_vectors = None
                    # task_inner_steps = to_int(inner_steps, args.inner_steps_min, args.inner_steps_max)
                    if args.use_MSL and step < args.MSL_iterations:
                        per_step_loss_importance_vectors = get_per_step_loss_importance_vector(
                                    inner_steps, args.MSL_iterations, step, device)
                    
                    
                    # https://github.com/tristandeleu/pytorch-meta/blob/master/examples/maml/train.py
                    outer_loss = torch.tensor(0.).to(device)
                    batch_size = args.MAML_batch
                    train_data = prepare_MAML_data(imgs, poses, batch_size, hwf, device)
                                        
                    # In MAML, the losses of a batch tasks were used to update meta parameter 
                    # but the batch of tasks in NeRF does not makes too much sense
                    # should it be a batch of scenes? or a batch of pixels in a single scene
                    for i in range(batch_size):
                        # update parameter with the inner loop loss
                        loss = inner_loop(meta_model, bound, args.num_samples,
                            args.train_batchsize, inner_steps, inner_lr, train_data[i],
                            per_step_loss_importance_vectors=per_step_loss_importance_vectors,
                            init_lr=args.inner_lr,
                            first_order= not (args.use_second_order and step>args.first_order_to_second_order_iteration))
                            
                        outer_loss += loss
                      
                    if args.per_layer_inner_lr: 
                        pbar.set_postfix({
                            'inner_lr': inner_lr['net.1.weight'][0].item(), 
                            "outer_lr" : scheduler.get_last_lr()[0], 
                            'Train loss': loss.item()
                        })
                    meta_optim.zero_grad()
                    outer_loss.div_(batch_size)
                    outer_loss.backward()
                
                    meta_optim.step()
                    
                    task_difficulty.append(outer_loss.item()) 
                    if step >= args.difficulty_size and args.learn_inner_step:
                        inner_steps, difficulty_rank = update_inner_steps(task_difficulty, args.inner_steps_min, args.inner_steps_max, args.difficulty_size)
                        if step % 10 == 0:
                            log_difficulty_rank.append(difficulty_rank)
                            log_inner_steps.append(inner_steps)
            
            else:
                meta_optim.zero_grad()
                
                inner_model = copy.deepcopy(meta_model)
                # inner_optim = torch.optim.SGD(inner_model.parameters(), args.inner_lr)
                
                inner_loop_Reptile(inner_model, imgs, poses,
                            hwf, bound, args.num_samples,
                            args.train_batchsize, args.inner_steps, inner_lr, args.inner_lr)
                            
                # Reptile
                with torch.no_grad():
                    for meta_param, inner_param in zip(meta_model.parameters(), inner_model.parameters()):
                        meta_param.grad = meta_param - inner_param
                        
                meta_optim.step()
                
                if args.learn_inner_lr:
                    lr_optim.zero_grad()
                    # loss.backward()
                    
                    for (name, param) in meta_model.meta_named_parameters():
                        # inner_lr[key] = args.inner_lr * torch.ones_like(param, requires_grad=True)
                        # inner_lr[key] = nn.Parameter(args.inner_lr * torch.ones_like(param, requires_grad=True))
                        # print(param.grad)
                        inner_lr[name].grad = param.grad
                        
                    lr_optim.step()
                    
                
        
            if step % args.val_freq == 0 and step != args.resume_step:
                # show one of the validation result
                test_psnrs = []
                if args.meta == 'MAML':
                    train_psnrs.append((step, -10*torch.log10(outer_loss).detach().cpu().item()))
                else:
                    meta_trained_state = meta_model.state_dict()
                    val_model = copy.deepcopy(meta_model)
                
                for imgs, poses, hwf, bound in tqdm(val_loader, desc = 'Validating'):
                    imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
                    imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
            
                    tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
                    tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)
                    
                    if args.meta == 'MAML':
                        scene_psnr = validate_MAML(meta_model, tto_imgs, tto_poses, test_imgs, test_poses, hwf, bound, inner_lr, args)            
                    else:
                        val_model.load_state_dict(meta_trained_state)
                        # val_optim = torch.optim.SGD(val_model.parameters(), args.tto_lr)
                        inner_loop_Reptile(val_model, tto_imgs, tto_poses, hwf,
                                    bound, args.num_samples, args.tto_batchsize, args.tto_steps, inner_lr, args.inner_lr)
                        
                        # only show imgs of two scenes
                        scene_psnr = report_result(model=val_model, test_imgs=test_imgs, test_poses=test_poses,
                                                    hwf=hwf, bound=bound, raybatch_size=args.test_batchsize, 
                                                    num_samples=args.num_samples, tto_showImages=args.tto_showImages,
                                                    params=None, show_img=len(test_psnrs)<args.show_validate_scene)
                        # scene_psnr = report_result(val_model, test_imgs, test_poses, hwf, bound, args.test_batchsize,
                        #                             args.num_samples, args.tto_showImages, None, count<=2)
                                                
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
                plt.savefig(f'{args.checkpoint_path}/{step}.png', bbox_inches='tight')
                plt.show()
                print(val_psnrs)
                
                if args.plot_lr:
                    if args.per_layer_inner_lr:
                        plt.subplots()
                        plt.ylabel("inner learning rate")
                        plt.xlabel("iterations")
                        plt.title("per param inner learning rate (first param of each layer)")
                        for (name, lrs) in inner_lrs.items():
                            plt.plot(lrs, label=name)
                        
                        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
                        
                        plt.savefig(f'{args.checkpoint_path}/{step}_lr.png', bbox_inches='tight')
                        plt.show()
                    
                    # ===================
                    
                        # plt.subplots()
                        # plt.ylabel("per step learning rate")
                        # plt.xlabel("iterations")
                        # plt.title(f"per layers per steps inner learning rate for layer {first_layer_name}")
                        # for (idx, lrs) in enumerate(inner_per_step_lr):
                        #     plt.plot(lrs, label=f"inner step {idx}")
                            
                        # if len(inner_per_step_lr) <= 8:
                        #     plt.legend()
                        # else:
                        #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        
                        # plt.savefig(f'{args.checkpoint_path}/{step}_per_steps_lr.png', bbox_inches='tight')
                        # plt.show()
                    
                    else:
                        plt.subplots()
                        plt.ylabel("inner learning rate")
                        plt.xlabel("iterations")
                        plt.title("inner learning rate")
                        plt.plot(inner_lrs, label="inner learning rate")
                        
                        plt.legend()
                        plt.savefig(f'{args.checkpoint_path}/{step}_lr.png', bbox_inches='tight')
                        plt.show()
                    # ===================
                    if args.learn_inner_step:
                        plt.subplots()
                        plt.ylabel("inner steps")
                        plt.xlabel("iterations")
                        plt.title(f"adaptive inner steps ({args.inner_steps_min} to {args.inner_steps_max})")
                        plt.plot(log_inner_steps)
                        plt.savefig(f'{args.checkpoint_path}/{step}_inner_steps.png', bbox_inches='tight')
                        plt.show()
                        
                        
                        plt.subplots()
                        plt.ylabel("difficulty_rank")
                        plt.xlabel("iterations")
                        plt.title(f"difficulty_rank (0 to {args.difficulty_size})")
                        plt.plot(log_difficulty_rank)
                        plt.savefig(f'{args.checkpoint_path}/{step}_difficulty_rank.png', bbox_inches='tight')
                        plt.show()
                    
                
                with open(f'{args.checkpoint_path}/psnr.txt', 'w') as f:
                    psnr = {
                        'train': train_psnrs,
                        'val' : val_psnrs,
                        'best_val': sorted(val_psnrs,key=lambda x: x[1], reverse=True)[0]
                    }
                    f.write(json.dumps(psnr))
                    
                with open(f'{args.checkpoint_path}/inner_lrs.txt', 'w') as f:
                    txt = {
                        'inner_lrs': inner_lrs,
                    }
                    f.write(json.dumps(txt))
                    
          
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
                if args.per_layer_inner_lr:
                    for (name, param) in meta_model.meta_named_parameters():
                        # print(name)
                        # print(inner_lr[name])
                        # inner_lrs[name].append(inner_lr[name][0][0])
                        if inner_lr[name].ndim == 1:
                            inner_lrs[name].append(inner_lr[name][0].item())
                        else:
                            inner_lrs[name].append(inner_lr[name][0][0].item())
                    # print(inner_lrs)
                    
                    # for idx, lr_list in enumerate(inner_per_step_lr):
                    #     lr_list.append(inner_lr[first_layer_name][idx].item())
                    
                else:
                    inner_lrs.append(inner_lr.item())
                    
            step += 1
            pbar.update(1)
            
            if step > args.max_iters:
              break
        
    pbar.close()

if __name__ == '__main__':
    main()