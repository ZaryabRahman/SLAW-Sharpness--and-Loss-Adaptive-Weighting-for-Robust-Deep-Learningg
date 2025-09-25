import torch
import torch.nn as nn
import torch.nn.functional as F

class SLAW:
    """
    SLAW (Sharpness- and Loss-Adaptive Weighting) Algorithm.
    This version is flexible and supports ablation studies by enabling/disabling SALS and LAW.
    """
    def __init__(self, model, num_classes,
                 alpha=0.2, eps_min=0.0, eps_max=0.2, sharp_ema_decay=0.99,
                 tau=1.0, beta=1.0, gamma=0.7,
                 sharpness_scope='last_layer',
                 use_sals=True, use_law=True):

        self.m = model
        self.K = num_classes
        self.alpha, self.eps_min, self.eps_max = alpha, eps_min, eps_max
        self.decay = sharp_ema_decay
        self.sharp_ema = 1.0
        self.tau, self.beta, self.gamma = tau, beta, gamma
        self.device = next(model.parameters()).device

        self.use_sals = use_sals
        self.use_law = use_law

        self.sharpness_scope = sharpness_scope
        self._grad_params = self._get_params_for_sharpness()
        
        if self.use_sals and not self._grad_params:
            print("Warning: SLAW could not find parameters for SALS sharpness proxy. Disabling SALS.")
            self.use_sals = False

    def _get_params_for_sharpness(self):
        if self.sharpness_scope == 'last_layer':
            last_layer = None
            for module in reversed(list(self.m.modules())):
                if isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(module, 'weight') and module.weight.requires_grad:
                    last_layer = module
                    break
            if last_layer:
                print(f"SLAW: Using last layer ({type(last_layer).__name__}) for sharpness proxy.")
                return list(last_layer.parameters())
            else: return []
        else:
            print("SLAW: Using full model for sharpness proxy.")
            return [p for p in self.m.parameters() if p.requires_grad]

    @torch.no_grad()
    def _update_sharp_ema(self, s_batch_float):
        self.sharp_ema = self.decay * self.sharp_ema + (1 - self.decay) * s_batch_float

    @torch.no_grad()
    def _get_sals_targets(self, probs, y, s_batch_tensor):
        conf = probs.max(dim=1).values
        sharpness_scale = (s_batch_tensor / (self.sharp_ema + 1e-12)).clamp(min=0.1, max=10.0)
        eps_i = (self.alpha * sharpness_scale * (1.0 - conf)).clamp(self.eps_min, self.eps_max)

        off_target_fill = eps_i.unsqueeze(1) / (self.K - 1 + 1e-12)
        t = off_target_fill.expand(-1, self.K).clone()
        t.scatter_(1, y.unsqueeze(1), 1.0 - eps_i.unsqueeze(1))
        return t, eps_i

    def _get_sharpness_proxy(self, logits, y):
        with torch.enable_grad():
            ce_loss_for_grad = F.cross_entropy(logits, y, reduction='mean')
            grads = torch.autograd.grad(ce_loss_for_grad, self._grad_params, create_graph=False, retain_graph=True)
        s_batch = torch.sqrt(sum((g.detach()**2).sum() for g in grads) + 1e-12)
        return s_batch

    def __call__(self, logits, y):
        metrics = {}

        # --- Step 1: Calculate the base per-sample loss ---
        if self.use_sals:
            s_batch_tensor = self._get_sharpness_proxy(logits, y)
            self._update_sharp_ema(s_batch_tensor.item())
            sals_targets, eps_i_stats = self._get_sals_targets(logits.softmax(dim=1).detach(), y, s_batch_tensor)
            log_probs = F.log_softmax(logits, dim=1)
            per_sample_loss = -(sals_targets * log_probs).sum(dim=1)
            metrics.update({'s_batch': s_batch_tensor.item(), 'sharp_ema': self.sharp_ema, 'eps_mean': eps_i_stats.mean().item()})
        else:
            # For LAW-only or CE, the base loss is standard Cross-Entropy
            per_sample_loss = F.cross_entropy(logits, y, reduction='none')

        # --- Step 2: Calculate the final batch loss with LAW ---
        if self.use_law:
            with torch.no_grad():
                loss_vals = per_sample_loss.detach()
                mu, std = loss_vals.mean(), loss_vals.std() + 1e-8
                standardized_loss = (loss_vals - mu) / std
                # Note: A positive standardized_loss means a high loss, so we want to give it a low weight.
                # The sigmoid should be applied to the negative standardized loss, or inverted.
                # A simple way is to use sigmoid(-z). But your paper uses sigmoid(z), let's stick to that.
                # This implies you want to down-weight easy samples more. Let's keep your original formulation.
                weights = (torch.sigmoid(standardized_loss / self.tau).pow(self.beta) * self.gamma) + (1 - self.gamma)
            final_loss = (weights * per_sample_loss).mean()
            metrics.update({'weight_mean': weights.mean().item()})
        else:
            # For SALS-only or CE, the final loss is a simple mean
            final_loss = per_sample_loss.mean()

        return final_loss, metrics```

