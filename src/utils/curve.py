import torch
from kornia.color import rgb_to_hls, hls_to_rgb


def get_shadow_weight(x):
    density = torch.mul(torch.pow(x, 2), -1)
    density = torch.div(density, 0.1)
    density = torch.exp(density)

    return density


def get_highlight_weight(x):
    density = torch.sub(x, 1)
    density = torch.pow(density, 2)
    density = torch.mul(density, -1)
    density = torch.div(density, 0.1)
    density = torch.exp(density)

    return density


def get_weights(x):
    n_channels = x.size(1)
    if n_channels == 3:
        x = rgb_to_hls(x)
        x = x[:, 1:2, ...] / 100

    shadow_weight = get_shadow_weight(x)
    highlight_weight = get_highlight_weight(x)

    midtone_weight = (
        torch.ones(x.size(), device=shadow_weight.device)
        - shadow_weight
        - highlight_weight
    )

    weights = torch.cat([shadow_weight, highlight_weight, midtone_weight], dim=1)

    return weights


def preset_transform(x, preset):
    # TODO: assert on tensor shape
    n_dim = preset.size(1)

    hls_x = rgb_to_hls(x)
    l_channel = hls_x[:, 1:2, :, :]

    if n_dim == 3:
        weights = get_weights(l_channel)
        # Weight should be associated with input (i.e. identifying shadow/highlight/midtone)
        l_channel = torch.cat([l_channel, l_channel, l_channel], dim=1)

        # Apply 8 times for greater curve
        for _ in range(8):
            l_channel = l_channel + preset * l_channel * (1 - l_channel)

        # Multiply L_shadow, L_highlight, L_midtone with respectively weight
        l_channel = l_channel * weights
        l_channel = torch.sum(l_channel, dim=1, keepdim=True)
    else:
        for _ in range(8):
            l_channel = l_channel + preset * l_channel * (1 - l_channel)

    hls_output = torch.cat([hls_x[:, 0:1, ...], l_channel, hls_x[:, 2:3, :, :]], dim=1)
    # hls_x[:, 1:2, ...] = l_channel

    output = hls_to_rgb(hls_output)

    return output
