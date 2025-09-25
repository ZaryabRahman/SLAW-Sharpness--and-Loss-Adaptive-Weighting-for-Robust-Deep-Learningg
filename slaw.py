import torch
import torch.nn as nn
import torch.nn.functional as F

class SLAW:
    """
    implements the SLAW (Sharpness- and Loss-Adaptive Weighting) algorithm.

    SLAW is a training algorithm for deep neural networks that enhances robustness by
    jointly addressing two key challenges: convergence to sharp minima and vulnerability
    to noisy labels. It integrates two adaptive mechanisms:
    
    1.  SALS (Sharpness-Adaptive Label Smoothing):   
                A regularizer that adjusts
                label smoothing intensity based on the local sharpness of the loss landscape.
    2.  LAW (Loss-Adaptive Reweighting):
                A statistical filter that down-weights
                the influence of likely mislabeled samples within each mini-batch.

    this class is designed to be a flexible loss function wrapper that can be used
    in place of `nn.CrossEntropyLoss`. It also supports ablation studies by allowing
    SALS and LAW components to be individually enabled or disabled.

    :param model: The neural network model being trained.
    :type model: nn.Module
    :param num_classes: The number of classes in the classification task.
    :type num_classes: int
    :param alpha: The base intensity factor for SALS. Controls the overall strength
                  of the adaptive label smoothing. Defaults to 0.2.
    :type alpha: float
    :param eps_min: The minimum value for the adaptive smoothing parameter epsilon.
                    Defaults to 0.0.
    :type eps_min: float
    :param eps_max: The maximum value for the adaptive smoothing parameter epsilon.
                    Acts as a cap on the regularization strength. Defaults to 0.2.
    :type eps_max: float
    :param sharp_ema_decay: The decay factor for the Exponential Moving Average (EMA)
                            of the sharpness proxy. Defaults to 0.99.
    :type sharp_ema_decay: float
    :param tau: The temperature parameter for the LAW sigmoid function. Controls the
                steepness of the reweighting curve. Defaults to 1.0.
    :type tau: float
    :param beta: The exponent for the LAW sigmoid function. Adjusts the shape of the
                 reweighting curve. Defaults to 1.0.
    :type beta: float
    :param gamma: The mixing coefficient for LAW. Blends the adaptive weight with a
                  uniform baseline, preventing weights from collapsing to zero.
                  Represents the maximum influence of the adaptive component.
                  Defaults to 0.7.
    :type gamma: float
    :param sharpness_scope: Defines which model parameters to use for the sharpness
                            proxy calculation ('last_layer' or 'full').
                            Defaults to 'last_layer'.
    :type sharpness_scope: str
    :param use_sals: Flag to enable the SALS component. Defaults to True.
    :type use_sals: bool
    :param use_law: Flag to enable the LAW component. Defaults to True.
    :type use_law: bool
    """
    def __init__(self, model: nn.Module, num_classes: int,
                 alpha: float = 0.2, eps_min: float = 0.0, eps_max: float = 0.2, sharp_ema_decay: float = 0.99,
                 tau: float = 1.0, beta: float = 1.0, gamma: float = 0.7,
                 sharpness_scope: str = 'last_layer',
                 use_sals: bool = True, use_law: bool = True):

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
        """
        identifies the model parameters to be used for the sharpness proxy calculation.

        based on the `sharpness_scope` attribute, this method returns either the
        parameters of the last trainable layer or all trainable parameters of the model.

        :return: A list of model parameters.
        :rtype: list[torch.Tensor]
        """
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
    def _update_sharp_ema(self, s_batch_float: float):
        """
        updates the Exponential Moving Average (EMA) of the batch sharpness.

        :param s_batch_float: The sharpness value (L2 norm of gradient) for the current batch.
        :type s_batch_float: float
        """
        self.sharp_ema = self.decay * self.sharp_ema + (1 - self.decay) * s_batch_float

    @torch.no_grad()
    def _get_sals_targets(self, probs: torch.Tensor, y: torch.Tensor, s_batch_tensor: torch.Tensor):
        """
        computes the adaptive smoothed target labels for each sample in a batch.

        The smoothing factor `epsilon_i` for each sample is determined by the model's
        confidence, the current batch sharpness, and the EMA of sharpness.

        :param probs: The softmax probability distribution over classes for the batch.
        :type probs: torch.Tensor
        :param y: The ground-truth labels for the batch.
        :type y: torch.Tensor
        :param s_batch_tensor: The sharpness proxy value for the current batch.
        :type s_batch_tensor: torch.Tensor
        :return: A tuple containing the smoothed target tensor and the per-sample epsilon values.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        conf = probs.max(dim=1).values
        sharpness_scale = (s_batch_tensor / (self.sharp_ema + 1e-12)).clamp(min=0.1, max=10.0)
        eps_i = (self.alpha * sharpness_scale * (1.0 - conf)).clamp(self.eps_min, self.eps_max)

        off_target_fill = eps_i.unsqueeze(1) / (self.K - 1 + 1e-12)
        t = off_target_fill.expand(-1, self.K).clone()
        t.scatter_(1, y.unsqueeze(1), 1.0 - eps_i.unsqueeze(1))
        return t, eps_i

    def _get_sharpness_proxy(self, logits: torch.Tensor, y: torch.Tensor):
        """
        calculates the sharpness proxy for the current mini-batch.

        the proxy is defined as the L2 norm of the gradient of the standard
        cross-entropy loss with respect to the parameters specified by `_grad_params`.

        :param logits: The model's output logits for the batch.
        :type logits: torch.Tensor
        :param y: The ground-truth labels for the batch.
        :type y: torch.Tensor
        :return: A tensor containing the scalar sharpness value.
        :rtype: torch.Tensor
        """
        with torch.enable_grad():
            ce_loss_for_grad = F.cross_entropy(logits, y, reduction='mean')
            # `retain_graph=True` is crucial as the main backward pass still needs the graph.
            grads = torch.autograd.grad(ce_loss_for_grad, self._grad_params, create_graph=False, retain_graph=True)
        s_batch = torch.sqrt(sum((g.detach()**2).sum() for g in grads) + 1e-12)
        return s_batch

    def __call__(self, logits: torch.Tensor, y: torch.Tensor):
        """
        computes the final SLAW loss for a given batch of logits and labels.

        this method orchestrates the SALS and LAW components based on their
        `use_sals` and `use_law` flags.

        :param logits: The model's output logits for the batch.
        :type logits: torch.Tensor
        :param y: The ground-truth labels for the batch.
        :type y: torch.Tensor
        :return: A tuple containing the final loss tensor and a dictionary of internal metrics.
        :rtype: tuple[torch.Tensor, dict]
        """
        metrics = {}

        #  calculate the base per-sample loss 
        if self.use_sals:
            s_batch_tensor = self._get_sharpness_proxy(logits, y)
            self._update_sharp_ema(s_batch_tensor.item())
            sals_targets, eps_i_stats = self._get_sals_targets(logits.softmax(dim=1).detach(), y, s_batch_tensor)
            log_probs = F.log_softmax(logits, dim=1)
            per_sample_loss = -(sals_targets * log_probs).sum(dim=1)
            metrics.update({'s_batch': s_batch_tensor.item(), 'sharp_ema': self.sharp_ema, 'eps_mean': eps_i_stats.mean().item()})
        else:
            # for LAW-only or standard CE, the base loss is just cross-entropy.
            per_sample_loss = F.cross_entropy(logits, y, reduction='none')

        # calculate the final batch loss with LAW 
        if self.use_law:
            with torch.no_grad():
                loss_vals = per_sample_loss.detach()
                mu, std = loss_vals.mean(), loss_vals.std() + 1e-8
                standardized_loss = (loss_vals - mu) / std
                weights = (torch.sigmoid(standardized_loss / self.tau).pow(self.beta) * self.gamma) + (1 - self.gamma)
            
            final_loss = (weights * per_sample_loss).mean()
            metrics.update({'weight_mean': weights.mean().item()})
        else:
            # for SALS-only or standard CE, the final loss is a simple mean.
            final_loss = per_sample_loss.mean()

        return final_loss, metrics
