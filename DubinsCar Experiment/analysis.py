import torch

def compute_spectral_norm(net):
    spectral_norms = {}

    for name, param in net.named_parameters():
        if 'weight' in name:

            spectral_norm = torch.svd(param.data).S.max(-1).values
            spectral_norms[name] = spectral_norm.item()
    return spectral_norms