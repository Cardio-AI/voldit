import torch
import torch.nn.functional as F
from monai.losses import PatchAdversarialLoss, PerceptualLoss, JukeboxLoss


class VQGANLoss:
    """
    Combined loss for VQGAN training: reconstruction + perceptual + adversarial + spectral + jukebox.
    """

    def __init__(
        self,
        perceptual_weight: float = 1.0,
        jukebox_weight: float = 1.0,
        patch_adv_params: dict | None = None,
        jukebox_params: dict | None = None,
        perceptual_params: dict | None = None,

        device: str = "cuda",
    ):
        self.perceptual_weight = perceptual_weight

        self.jukebox_weight = jukebox_weight

        # Initialize optional MONAI loss modules only when they contribute.
        self.perceptual_loss = (
            PerceptualLoss(**(perceptual_params or {})).to(device)
            if self.perceptual_weight > 0
            else None
        )
        self.adversarial_loss = PatchAdversarialLoss().to(device)
        self.jukebox_loss = (
            JukeboxLoss(**(jukebox_params or {})).to(device)
            if self.jukebox_weight > 0
            else None
        )

    def generator_loss(
        self,
        discriminator,
        adv_weight,
        images: torch.Tensor,
        reconstructions: torch.Tensor,
        quantization_loss: torch.Tensor,
    ):
        """
        Compute the generator loss combining:
        L1 reconstruction + perceptual + spectral + patch adversarial + jukebox loss
        """

        # Reconstruction L1
        l1 = F.l1_loss(reconstructions, images)

        p = (
            self.perceptual_loss(reconstructions.float(), images.float())
            if self.perceptual_loss is not None
            else torch.zeros_like(l1)
        )

        # Adversarial loss (patch-based)
        if adv_weight > 0:
            disc_out = discriminator(reconstructions.contiguous())
            logits_fake = disc_out[-1]
            g_adv = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
        else:
            g_adv = torch.zeros_like(l1)

        j_loss = (
            self.jukebox_loss(reconstructions.float(), images.float())
            if self.jukebox_loss is not None
            else torch.zeros_like(l1)
        )

        total_loss = (
            l1.float()
            + quantization_loss.float()
            + self.perceptual_weight * p
            + adv_weight * g_adv.float()
            + self.jukebox_weight * j_loss
        )

        logs = {
            "l1_loss": l1.mean(),
            "quantization_loss": quantization_loss.mean(),
            "perceptual_loss": p.mean(),
            "g_adv_loss": g_adv.mean(),
            "jukebox_loss": j_loss.mean(),
        }

        return total_loss.mean(), logs


    def discriminator_loss(self, discriminator, adv_weight, images: torch.Tensor, reconstructions: torch.Tensor):
        """
        Patch adversarial discriminator loss for real and fake images
        """

        # Fake images
        disc_out_fake = discriminator(reconstructions.detach())
        logits_fake = disc_out_fake[-1]
        loss_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)

        # Real images
        disc_out_real = discriminator(images.detach())
        logits_real = disc_out_real[-1]
        loss_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
        d_loss = 0.5 * (loss_fake + loss_real).mean()
        return d_loss, {"d_loss": d_loss}
