import torch


class FSAM(torch.optim.Optimizer):
    def __init__(self, named_params, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        named_params = list(named_params)
        names = [named_param[0] for named_param in named_params]
        defaults = dict(names=names, rho=rho, adaptive=adaptive, **kwargs)
        super(FSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm_mask, grad_norm_denoise, grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale_mask = group["rho"] / (grad_norm_mask + 1e-12)
            scale_denoise = group["rho"] / (grad_norm_denoise + 1e-12)
            scale = group["rho"] / (grad_norm + 1e-12)

            for p, n in zip(group["params"], group["names"]):
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["name"] = n
                if p.grad is None:
                    continue
                if 'encoder' in n:
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    self.state[p]["e_w"] = e_w
                    p.add_(e_w)
                elif 'mask_decoder' in n:
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale_mask.to(p)
                    self.state[p]["e_w_mask"] = e_w
                    p.add_(e_w)
                elif 'denoise_decoder' in n:
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale_denoise.to(p)
                    self.state[p]["e_w_denoise"] = e_w
                    p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        temp_mask_list = []
        temp_denoise_list = []
        temp_list = []
        for group in self.param_groups:
            for p, n in zip(group["params"], group["names"]):
                if p.grad is not None:
                    if group["adaptive"]:
                        temp_list.append((torch.abs(p) * p.grad).norm(p=2).to(shared_device))
                        if 'encoder' in n:
                            temp_mask_list.append((torch.abs(p) * p.grad).norm(p=2).to(shared_device))
                            temp_denoise_list.append((torch.abs(p) * p.grad).norm(p=2).to(shared_device))
                        elif 'mask_decoder' in n:
                            temp_mask_list.append((torch.abs(p) * p.grad).norm(p=2).to(shared_device))
                        elif 'denoise_decoder' in n:
                            temp_denoise_list.append((torch.abs(p) * p.grad).norm(p=2).to(shared_device))
                    else:
                        temp_list.append(p.grad.norm(p=2).to(shared_device))
                        if 'encoder' in n:
                            temp_mask_list.append(p.grad.norm(p=2).to(shared_device))
                            temp_denoise_list.append(p.grad.norm(p=2).to(shared_device))
                        elif 'mask_decoder' in n:
                            temp_mask_list.append(p.grad.norm(p=2).to(shared_device))
                        elif 'denoise_decoder' in n:
                            temp_denoise_list.append(p.grad.norm(p=2).to(shared_device))

        temp_mask = torch.stack(temp_mask_list)
        temp_denoise = torch.stack(temp_denoise_list)
        temp = torch.stack(temp_list)

        norm_mask = torch.norm(temp_mask, p=2)
        norm_denoise = torch.norm(temp_denoise, p=2)
        norm = torch.norm(temp, p=2)
        return norm_mask, norm_denoise, norm

