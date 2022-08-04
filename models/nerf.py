import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear


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
        sigma = F.softplus(out[..., -1])
        return rgb, sigma


def build_nerf(args):
    model = MetaSimpleNeRF(in_features=3, max_freq=args.max_freq, num_freqs=args.num_freqs,
                    hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
                    out_features=4)
    return model