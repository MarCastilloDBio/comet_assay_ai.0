"""
Modelo U-Net con encoder pre-entrenado para segmentación de cometa
Usa segmentation_models_pytorch para facilitar arquitectura
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional


class UNetComet(nn.Module):
    """
    U-Net con encoder ResNet34 pre-entrenado para segmentación 3-clase

    Args:
        encoder_name: Nombre del encoder ('resnet34', 'resnet50', 'efficientnet-b0', etc.)
        encoder_weights: Pesos pre-entrenados ('imagenet' o None)
        in_channels: Canales de entrada (3 para RGB)
        classes: Número de clases (3: fondo, cabeza, cola)
    """

    def __init__(
            self,
            encoder_name: str = 'resnet34',
            encoder_weights: Optional[str] = 'imagenet',
            in_channels: int = 3,
            classes: int = 3
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None  # No aplicamos softmax aquí (lo hace la loss)
        )

        self.encoder_name = encoder_name
        self.classes = classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Tensor [B, C, H, W]

        Returns:
            logits: Tensor [B, num_classes, H, W]
        """
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicción con softmax aplicado

        Returns:
            probs: Tensor [B, num_classes, H, W] con probabilidades
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicción de máscara (clase con máxima probabilidad)

        Returns:
            mask: Tensor [B, H, W] con valores 0, 1, 2
        """
        probs = self.predict(x)
        return torch.argmax(probs, dim=1)


class DiceLoss(nn.Module):
    """
    Dice Loss multi-clase para segmentación

    Args:
        smooth: Factor de suavizado para evitar división por cero
        ignore_index: Clase a ignorar (típicamente fondo=0)
    """

    def __init__(self, smooth: float = 1.0, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] - salida del modelo
            targets: [B, H, W] - ground truth (valores enteros)
        """
        # Convertir logits a probabilidades
        probs = F.softmax(logits, dim=1)

        # One-hot encoding de targets
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Calcular Dice por clase
        dice_per_class = []
        for c in range(logits.shape[1]):
            if self.ignore_index is not None and c == self.ignore_index:
                continue

            pred_c = probs[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice)

        # Promedio de Dice por clase
        dice_loss = 1.0 - torch.stack(dice_per_class).mean()

        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combinación de CrossEntropy + Dice Loss

    Args:
        ce_weight: Peso para CrossEntropy (default 0.5)
        dice_weight: Peso para Dice Loss (default 0.5)
        class_weights: Pesos por clase para CrossEntropy (opcional)
    """

    def __init__(
            self,
            ce_weight: float = 0.5,
            dice_weight: float = 0.5,
            class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss(smooth=1.0, ignore_index=None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W]
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total_loss = self.ce_weight * ce + self.dice_weight * dice

        return total_loss


# Test del modelo
if __name__ == "__main__":
    # Crear modelo
    model = UNetComet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=3
    )

    print(model)
    print(f"\nTotal de parámetros: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512)

    with torch.no_grad():
        logits = model(x)
        probs = model.predict(x)
        masks = model.predict_mask(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Valores únicos en máscara: {torch.unique(masks)}")

    # Test loss
    targets = torch.randint(0, 3, (batch_size, 512, 512))
    loss_fn = CombinedLoss()
    loss = loss_fn(logits, targets)
    print(f"\nLoss: {loss.item():.4f}")