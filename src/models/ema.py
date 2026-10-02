import math
import torch


class EMA:
    """
    Exponential Moving Average of model parameters.

    When ``warmup_steps > 0`` the effective decay ramps from ~0.9 up to the
    target ``decay`` over that many update steps using a cosine schedule.
    This prevents the EMA from averaging over many poor early checkpoints,
    which would otherwise slow down convergence of the EMA model.
    """

    def __init__(self, model, decay: float = 0.9999, warmup_steps: int = 0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        self.model = model
        self.shadow: dict = {}
        self.backup: dict = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _effective_decay(self) -> float:
        """Return the decay to use at the current step."""
        if self.warmup_steps <= 0:
            return self.decay
        # Cosine ramp from 0.9 → self.decay over warmup_steps.
        if self.step >= self.warmup_steps:
            return self.decay
        progress = self.step / self.warmup_steps
        cosine = 0.5 * (1.0 - math.cos(math.pi * progress))   # 0 → 1
        start = 0.9
        return start + (self.decay - start) * cosine

    @torch.no_grad()
    def update(self):
        decay = self._effective_decay()
        self.step += 1
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    decay * self.shadow[name] + (1.0 - decay) * param.data
                ).clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "step": self.step,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.warmup_steps = state_dict.get("warmup_steps", 0)
        self.step = state_dict.get("step", 0)
        device = next(self.model.parameters()).device
        self.shadow = {k: v.to(device) for k, v in state_dict["shadow"].items()}
