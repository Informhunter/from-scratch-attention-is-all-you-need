import copy
from typing import Any, Callable, Dict, List
from enum import Enum

import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class ModelSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_1 = torch.nn.Parameter(torch.FloatTensor([7.]))
        self.param_2 = torch.nn.Parameter(torch.FloatTensor([8.]))
        self.linear = nn.Linear(100, 200)

    def forward(self):
        return self.param_1 + self.param_2


class ModelBig(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.FloatTensor([2.]))
        self.submodule = ModelSmall()

    def forward(self):
        return self.submodule() * self.param


class OptType(str, Enum):
    ADAM = 'adam'
    SGD = 'SGD'


class GradInit:
    def __init__(
            self,
            n_steps: int,
            gradinit_gamma: float,
            gradinit_lr: float,
            gradinit_alpha: float,
            target_opt_type: OptType,
            target_opt_params: Dict[str, Any],
    ):
        self.original_getattrs = {}
        self.state_backup = None

        self.n_steps = n_steps
        self.gradinit_gamma = gradinit_gamma
        self.gradinit_lr = gradinit_lr
        self.gradinit_alpha = gradinit_alpha

        self.target_opt_type = target_opt_type
        self.target_opt_params = target_opt_params

    def gradinit(
            self,
            module: nn.Module,
            dataloader: DataLoader,
            loss_fn: Callable[[nn.Module, "Batch"], torch.FloatTensor],
            cat_batch_fn: Callable[["Batch", "Batch"], "Batch"]
    ):
        self.monkeypatch_module_(module)

        data_iter = iter(dataloader)
        gradinit_opt = torch.optim.SGD(module.parameters(), lr=self.gradinit_lr)

        if self.target_opt_type == OptType.ADAM:
            gradient_norm_fn = self.gradient_norm_adam
            target_opt = torch.optim.Adam(module.parameters(), **self.target_opt_params)

        elif self.target_opt_type == OptType.SGD:
            gradient_norm_fn = self.gradient_norm_sgd
            target_opt = torch.optim.SGD(module.parameters(), **self.target_opt_params)
        else:
            raise NotImplementedError('Only support Adam and SGD currently')

        # Optimize scaling factors
        for i in range(self.n_steps):
            batch_1 = next(data_iter)

            # now calculating gradient with respect to model parameters
            module.zero_grad()
            self.set_scale_requires_grad_(module, False)
            self.set_non_scale_requires_grad_(module, True)
            loss_1 = loss_fn(module, batch_1)
            loss_1.backward()
            gradients = [param.grad for param in module.parameters() if param.requires_grad]

            gradient_norm = gradient_norm_fn(gradients)

            if gradient_norm > self.gradinit_gamma:
                # Calculate gradient with respect to scale factors and update them
                module.zero_grad()
                self.set_scale_requires_grad_(module, True)
                self.set_non_scale_requires_grad_(module, False)
                loss_1 = loss_fn(module, batch_1)
                loss_1.backward()
                gradinit_opt.step()
            else:
                self.backup_state(module)

                # Perform first training step using algorithm of choice (updates only module weights)
                target_opt.step()

                module.zero_grad()
                self.set_scale_requires_grad_(module, True)
                self.set_non_scale_requires_grad_(module, False)

                batch_2 = cat_batch_fn(batch_1, next(data_iter))

                loss_2 = loss_fn(module, batch_2)
                loss_2.backward()
                gradinit_opt.step()

                self.recover_state_(module)

        # Scale weights by scaling factors
        self.scale_weights_(module)

        self.cleanup_module_(module)

    def monkeypatch_module_(self, module: nn.Module):
        self.init_scale_parameters_(module)
        self.patch_getattr_(module)

    def cleanup_module_(self, module: nn.Module):
        self.unpatch_getattr_(module)
        self.remove_scale_parameters_(module)

    def init_scale_parameters_(self, module: nn.Module):
        for name, param in list(module._parameters.items()):
            setattr(module, name + '__scale', nn.Parameter(torch.FloatTensor([1.])))
        for name, submodule in list(module._modules.items()):
            self.init_scale_parameters_(submodule)

    def patch_getattr_(self, module: nn.Module):
        module_type = type(module)
        if module_type in self.original_getattrs:
            return
        self.original_getattrs[module_type] = module_type.__getattr__
        module_type.__getattr__ = self.getattr_patch

        for name, submodule in list(module._modules.items()):
            self.patch_getattr_(submodule)

    def safe_getattr(self, obj, key: str):
        module_type = type(obj)
        if module_type in self.original_getattrs:
            return self.original_getattrs[module_type](obj, key)
        else:
            return getattr(obj, key)

    def getattr_patch(self, obj, key: str):
        result = self.safe_getattr(obj, key)
        if isinstance(result, nn.Parameter):
            scale_factor = self.safe_getattr(obj, key + '__scale')
            return result * scale_factor
        else:
            return result

    def unpatch_getattr_(self, module: nn.Module):
        module_type = type(module)
        if module_type in self.original_getattrs:
            module_type.__getattr__ = self.original_getattrs[module_type]
        for name, submodule in list(module._modules.items()):
            self.unpatch_getattr_(submodule)

    def remove_scale_parameters_(self, module: nn.Module):
        for name, param in list(module._parameters.items()):
            if name.endswith('__scale'):
                delattr(module, name)
        for name, submodule in list(module._modules.items()):
            self.remove_scale_parameters_(submodule)

    def set_scale_requires_grad_(self, module: nn.Module, requires_grad: bool):
        for name, param in list(module.named_parameters()):
            if name.endswith('__scale'):
                param.requires_grad = requires_grad

    def set_non_scale_requires_grad_(self, module: nn.Module, requires_grad: bool):
        for name, param in list(module.named_parameters()):
            if not name.endswith('__scale'):
                param.requires_grad = requires_grad

    def backup_state(self, module: nn.Module):
        state = copy.deepcopy(module.state_dict())
        state = {name: param for name, param in state.items() if not name.endswith('__scale')}
        self.state_backup = state

    def recover_state_(self, module: nn.Module):
        module.load_state_dict(self.state_backup, strict=False)

    def scale_weights_(self, module: nn.Module):
        with torch.no_grad():
            for name, param in list(module._parameters.items()):
                if not name.endswith('__scale'):
                    scale_param = self.safe_getattr(name + '__scale')
                    param.copy_(param * scale_param)
        for name, submodule in list(module._modules.items()):
            self.scale_weights_(submodule)

    def clamp_scaling_factors_(self, module: nn.Module):
        with torch.no_grad():
            for name, param in list(module._parameters.items()):
                if name.endswith('__scale'):
                    scale_param = self.safe_getattr(name + '__scale')
                    clamped_value = torch.clamp(scale_param, min=self.gradinit_alpha)
                    scale_param.copy_(torch.FloatTensor([clamped_value]))

        for name, submodule in list(module._modules.items()):
            self.clamp_scaling_factors_(submodule)

    def gradient_norm_adam(self, gradients: List[torch.FloatTensor]) -> torch.FloatTensor:
        concatenated_gradients = torch.cat([gradient.flatten() for gradient in gradients])
        return torch.linalg.vector_norm(concatenated_gradients, 1.)

    def gradient_norm_sgd(self, gradients: List[torch.FloatTensor]) -> torch.FloatTensor:
        concatenated_gradients = torch.cat([gradient.flatten() for gradient in gradients])
        return torch.linalg.vector_norm(concatenated_gradients, 2.)


def main():
    model = ModelBig()
    g = GradInit()
    g.init_scale_parameters_(model)

    g.backup_state(model)

    with torch.no_grad():
        model.param__scale.copy_(torch.FloatTensor([55.]))

    with torch.no_grad():
        model.param.copy_(torch.FloatTensor([77.]))

    print('param before recovery:', model.param)
    print('param__scale before recovery:', model.param__scale)

    g.recover_state_(model)

    print('param after recovery:', model.param)
    print('param__scale after recovery:', model.param__scale)

    pass



if __name__ == '__main__':
    main()