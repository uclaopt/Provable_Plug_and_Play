"""
Spectral Normalization borrowed from https://arxiv.org/abs/1802.05957
SN for batch normalization layers to be of Lipschtz constant sigma (default=1.0).
"""
import torch
from torch.nn.parameter import Parameter


class BatchNormSpectralNorm(object):

    def __init__(self, name='weight', sigma=1.0, eps=1e-12):
        self.name = name
        self.sigma = sigma
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        bias = getattr(module, "bias_orig")
        running_var = getattr(module, "running_var")

        with torch.no_grad():
            cur_sigma = torch.max(torch.abs(weight / torch.sqrt(running_var)))
            # print(cur_sigma)
            cur_sigma = max(float(cur_sigma.cpu().detach().numpy()), self.sigma)
            # print(cur_sigma)

        weight = weight / cur_sigma
        bias = bias / cur_sigma
        return weight, bias

    def remove(self, module):
        weight = getattr(module, self.name)
        bias = getattr(module, "bias")
        delattr(module, self.name)
        delattr(module, self.name + '_orig')
        delattr(module, "bias")
        delattr(module, "bias_orig")
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))
        module.register_parameter("bias", torch.nn.Parameter(bias.detach()))

    def __call__(self, module, inputs):
        if module.training:
            weight, bias = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, "bias", bias)
        else:
            weight_r_g = getattr(module, self.name + '_orig').requires_grad
            bias_r_g = getattr(module, "bias_orig").requires_grad
            getattr(module, self.name).detach_().requires_grad_(weight_r_g)
            getattr(module, "bias").detach_().requires_grad_(bias_r_g)

    @staticmethod
    def apply(module, name, sigma, eps):
        fn = BatchNormSpectralNorm(name, sigma, eps)
        weight = module._parameters[name]
        bias = module._parameters["bias"]

        delattr(module, fn.name)
        delattr(module, "bias")
        module.register_parameter(fn.name + "_orig", weight)
        module.register_parameter("bias_orig", bias)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # buffer, which will cause weight to be included in the state dict
        # and also supports nn.init due to shared storage.
        module.register_buffer(fn.name, weight.data)
        module.register_buffer("bias", bias.data)

        module.register_forward_pre_hook(fn)
        return fn


def bn_spectral_norm(module, name='weight', sigma=1.0, eps=1e-12):
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
         \mathbf{W} &= \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) &= \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        eps (float, optional): epsilon for numerical stability in
            calculating norms

    Returns:
        The original module with the spectal norm hook

    Example::

        >>> m = batchnorm_spectral_norm(nn.BatchNorm2d(10))
        BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> m.weight_orig.size()
        torch.Size([10])

    """
    BatchNormSpectralNorm.apply(module, name, sigma, eps)
    return module


def remove_bn_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BatchNormSpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(
        name, module))
