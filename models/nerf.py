import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional Encoding of the input coordinates.
    
    encodes x to (..., sin(2^k x), cos(2^k x), ...)
    k takes "num_freqs" number of values equally spaced between [0, max_freq]
    """
    def __init__(self, max_freq, num_freqs):
        """
        Args:
            max_freq (int): maximum frequency in the positional encoding.
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()
        freqs = 2**torch.linspace(0, max_freq, num_freqs)
        self.register_buffer("freqs", freqs) #(num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (num_rays, num_samples, in_features)
        Outputs:
            out: (num_rays, num_samples, 2*num_freqs*in_features)
        """
        x_proj = x.unsqueeze(dim=-2)*self.freqs.unsqueeze(dim=-1) #(num_rays, num_samples, num_freqs, in_features)
        x_proj = x_proj.reshape(*x.shape[:-1], -1) #(num_rays, num_samples, num_freqs*in_features)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) #(num_rays, num_samples, 2*num_freqs*in_features)
        return out
 
class MetaSimpleNeRF(MetaModule):
    """
    A simple NeRF MLP without view dependence and skip connections.
    Implemented with MetaModule
    """
    def __init__(self, in_features, max_freq, num_freqs,
                 hidden_features, hidden_layers, out_features):
        """
        in_features: number of features in the input.
        max_freq: maximum frequency in the positional encoding.
        num_freqs: number of frequencies between [0, max_freq] in the positional encoding.
        hidden_features: number of features in the hidden layers of the MLP.
        hidden_layers: number of hidden layers.
        out_features: number of features in the output.
        """
        super().__init__()

        self.net = []
        self.net.append(PositionalEncoding(max_freq, num_freqs))
        
        self.net.append(MetaLinear(2*num_freqs*in_features, hidden_features))
        self.net.append(nn.ReLU())
    
        for i in range(hidden_layers-1):
            self.net.append(MetaLinear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())

        self.net.append(MetaLinear(hidden_features, out_features))
        self.net = MetaSequential(*self.net)
        # print(self.net)

    def forward(self, x, params=None):
        """
        At each input xyz point return the rgb and sigma values.
        Input:
            x: (num_rays, num_samples, 3)
        Output:
            rgb: (num_rays, num_samples, 3)
            sigma: (num_rays, num_samples)
        """
        out = self.net(x, params=self.get_subdict(params, 'net'))
        rgb = torch.sigmoid(out[..., :-1])
        # https://github.com/sanowar-raihan/nerf-meta/issues/3
        sigma = F.softplus(out[..., -1])
        # original paper use relu here
        # sigma = F.relu(out[..., -1])
        return rgb, sigma


def build_nerf(args):
    model = MetaSimpleNeRF(in_features=3, max_freq=args.max_freq, num_freqs=args.num_freqs,
                    hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
                    out_features=4)
    return model
    

import tinycudann as tcnn
import vren
from einops import rearrange

NEAR_DISTANCE = 0.01

from torch.cuda.amp import custom_fwd, custom_bwd
class TruncExp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))

class MetaNGP(MetaModule):
    def __init__(self, scale, rgb_act='Sigmoid'):
        super().__init__()

        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=16,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

        self.xyz_encoder = MetaSequential(self.xyz_encoder)
        
        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
        self.dir_encoder = MetaSequential(self.dir_encoder)
        
        self.rgb_net = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": rgb_act,
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )
        self.rgb_net = MetaSequential(self.rgb_net)
                

    def density(self, x, return_feat=False, params=None):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        h = self.xyz_encoder(x, params=self.get_subdict(params, 'xyz_encoder'))
        
        sigmas = TruncExp.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def forward(self, x, d=None, params=None):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, h = self.density(x, return_feat=True)
        if d is not None:
            d = d/torch.norm(d, dim=1, keepdim=True)
            d = self.dir_encoder((d+1)/2, params=self.get_subdict(params, 'dir_encoder'))
            rgbs = self.rgb_net(torch.cat([d, h], 1), params=self.get_subdict(params, 'rgb_net'))
        else:
            rgbs = self.rgb_net(h, params=self.get_subdict(params, 'rgb_net'))

        return sigmas, rgbs

    # Only update grids during TTO
    
    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a') # (N_cams, 3, 3)
        w2c_T = -w2c_R@poses[:, :3, 3:] # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i+chunk]/(self.grid_size-1)*2-1
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (xyzs*(s-half_grid_size)).T # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T # (N_cams, 3, chunk)
                uvd = K @ xyzs_c # (N_cams, 3, chunk)
                uv = uvd[:, :2]/uvd[:, 2:] # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2]>=NEAR_DISTANCE)&in_image # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i+chunk]] = \
                    count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2]<NEAR_DISTANCE)&in_image # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count>0)&(~too_near_to_any_cam)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup: # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.scale)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid>0].mean().item()

        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)
