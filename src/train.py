"""
Script de entrenamiento para modelo U-Net de segmentación de cometa
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from dataset_grayscale import CometDatasetGrayscale as CometDataset, get_train_transform_grayscale as get_train_transform, get_val_transform_grayscale as get_val_transform
from model import UNetComet, CombinedLoss
from metrics import dice_score, iou_score

# FIX PARA SSL EN WINDOWS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class Trainer:
    """
    Clase para gestionar el entrenamiento del modelo
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        output_dir: str,
        num_epochs: int = 100,
        early_stopping_patience: int = 15
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Historial de entrenamiento
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'lr': []
        }

        self.best_val_dice = 0.0
        self.epochs_without_improvement = 0

    def train_epoch(self) -> float:
        """Entrena una época"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    def validate(self) -> dict:
        """Valida el modelo"""
        self.model.eval()
        total_loss = 0.0
        dice_scores = {'head': [], 'tail': [], 'mean': []}
        iou_scores = {'head': [], 'tail': [], 'mean': []}

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Predicción
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                total_loss += loss.item()

                # Calcular métricas
                pred_masks = torch.argmax(logits, dim=1)

                for i in range(images.size(0)):
                    pred = pred_masks[i].cpu().numpy()
                    target = masks[i].cpu().numpy()

                    # Dice y IoU por clase
                    dice_head = dice_score(pred == 1, target == 1)
                    dice_tail = dice_score(pred == 2, target == 2)
                    iou_head = iou_score(pred == 1, target == 1)
                    iou_tail = iou_score(pred == 2, target == 2)

                    dice_scores['head'].append(dice_head)
                    dice_scores['tail'].append(dice_tail)
                    dice_scores['mean'].append((dice_head + dice_tail) / 2)

                    iou_scores['head'].append(iou_head)
                    iou_scores['tail'].append(iou_tail)
                    iou_scores['mean'].append((iou_head + iou_tail) / 2)

        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'dice_head': np.mean(dice_scores['head']),
            'dice_tail': np.mean(dice_scores['tail']),
            'dice_mean': np.mean(dice_scores['mean']),
            'iou_head': np.mean(iou_scores['head']),
            'iou_tail': np.mean(iou_scores['tail']),
            'iou_mean': np.mean(iou_scores['mean'])
        }

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Guarda checkpoint del modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_dice': self.best_val_dice
        }

        # Guardar último checkpoint
        torch.save(checkpoint, self.output_dir / 'last_checkpoint.pth')

        # Guardar mejor modelo
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            print(f"✓ Mejor modelo guardado (Dice: {self.best_val_dice:.4f})")

    def train(self):
        """Loop principal de entrenamiento"""
        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 60)

            # Entrenar
            train_loss = self.train_epoch()

            # Validar
            val_metrics = self.validate()

            # Actualizar scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_dice'].append(val_metrics['dice_mean'])
            self.history['val_iou'].append(val_metrics['iou_mean'])
            self.history['lr'].append(current_lr)

            # Imprimir métricas
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_metrics['val_loss']:.4f}")
            print(f"Dice Head:  {val_metrics['dice_head']:.4f}")
            print(f"Dice Tail:  {val_metrics['dice_tail']:.4f}")
            print(f"Dice Mean:  {val_metrics['dice_mean']:.4f}")
            print(f"IoU Mean:   {val_metrics['iou_mean']:.4f}")
            print(f"LR:         {current_lr:.6f}")

            # Early stopping y checkpoint
            is_best = False
            if val_metrics['dice_mean'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice_mean']
                self.epochs_without_improvement = 0
                is_best = True
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch, is_best=is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping activado (sin mejora en {self.early_stopping_patience} epochs)")
                break

        # Guardar historial
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Entrenamiento completado")
        print(f"Mejor Dice: {self.best_val_dice:.4f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo U-Net para segmentación de cometa')

    # Datos
    parser.add_argument('--data_dir', type=str, default='dataset', help='Directorio con dataset')
    parser.add_argument('--val_split', type=float, default=0.2, help='Proporción de validación')
    parser.add_argument('--image_size', type=int, default=512, help='Tamaño de imagen')

    # Modelo
    parser.add_argument('--encoder', type=str, default='resnet34',
                       help='Encoder (resnet34, resnet50, efficientnet-b0)')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                       help='Pesos pre-entrenados (imagenet o None)')

    # Entrenamiento
    parser.add_argument('--epochs', type=int, default=100, help='Número de epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--early_stopping', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--augmentation_prob', type=float, default=0.5, help='Probabilidad de augmentation')

    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directorio de salida')
    parser.add_argument('--resume', type=str, default=None, help='Path a checkpoint para continuar')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")

    # Crear directorios
    output_dir = Path(args.output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar configuración
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Dataset
    print("\nCargando dataset...")
    full_dataset = CometDataset(
        image_dir=os.path.join(args.data_dir, 'images'),
        mask_dir=os.path.join(args.data_dir, 'masks'),
        transform=None,
        image_size=(args.image_size, args.image_size)
    )

    # Split train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Aplicar transforms
    train_dataset.dataset.transform = get_train_transform(
        image_size=(args.image_size, args.image_size),
        p=args.augmentation_prob
    )
    val_dataset.dataset.transform = get_val_transform(
        image_size=(args.image_size, args.image_size)
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 0 para Windows
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 0 para Windows
        pin_memory=False
    )

    # Modelo
    print("\nCreando modelo...")
    model = UNetComet(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights if args.encoder_weights != 'None' else None,
        in_channels=3,
        classes=3
    ).to(device)

    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Resume desde checkpoint
    start_epoch = 0
    if args.resume:
        print(f"\nCargando checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Continuando desde epoch {start_epoch}")

    # Entrenar
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping
    )

    trainer.train()

    print(f"\n✓ Checkpoints guardados en: {output_dir}")


if __name__ == "__main__":
    main()