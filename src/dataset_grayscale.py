"""
Dataset modificado para trabajar SOLO con escala de grises
Más robusto para imágenes con diferentes fluoróforos
"""

import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CometDatasetGrayscale(Dataset):
    """
    Dataset para imágenes de cometa - VERSION ESCALA DE GRISES

    Convierte todas las imágenes a escala de grises para:
    - Robustez a diferentes fluoróforos
    - Compatibilidad con diferentes pseudo-colores
    - Enfoque en intensidad, no en color

    Args:
        image_dir: Directorio con imágenes originales
        mask_dir: Directorio con máscaras (0=fondo, 1=cabeza, 2=cola)
        transform: Transformaciones de albumentations (opcional)
        image_size: Tamaño de redimensión (altura, ancho)
        normalize: Si True, normaliza imágenes
    """

    def __init__(
            self,
            image_dir: str,
            mask_dir: str,
            transform: Optional[Callable] = None,
            image_size: Tuple[int, int] = (512, 512),
            normalize: bool = True
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.normalize = normalize

        # Listar archivos
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])

        print(f"Dataset inicializado: {len(self.image_files)} imágenes (modo escala de grises)")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna imagen y máscara procesadas

        Returns:
            image: Tensor [3, H, W] en escala de grises (3 canales idénticos)
            mask: Tensor [H, W] con valores 0, 1, 2
        """
        # Cargar imagen
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"No se pudo cargar imagen: {img_path}")

        # ========================================
        # CONVERSIÓN A ESCALA DE GRISES (CRÍTICO)
        # ========================================
        if len(image.shape) == 3:
            # Si es color (RGB/BGR/RGBA), convertir a grises
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:  # RGB o BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Si ya es gris (len(image.shape)==2), no hacer nada

        # Convertir escala de grises a 3 canales RGB (requisito del modelo)
        # Los 3 canales tendrán valores idénticos: [G, G, G]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Cargar máscara
        mask_name = img_name.replace('.tif', '.png').replace('.tiff', '.png').replace('.jpg', '.png').replace('.jpeg',
                                                                                                              '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            raise ValueError(f"Máscara no encontrada: {mask_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # ========================================
        # CONVERSIÓN A TIPO CORRECTO (CRÍTICO)
        # ========================================
        mask = mask.astype(np.int64)  # ← LÍNEA CRÍTICA AÑADIDA

        # Validar valores de máscara (deben ser 0, 1 o 2)
        unique_vals = np.unique(mask)
        if not np.all(np.isin(unique_vals, [0, 1, 2])):
            print(f"Advertencia: Máscara {mask_name} tiene valores inesperados: {unique_vals}")
            mask = np.clip(mask, 0, 2)

        # Redimensionar
        image_rgb = cv2.resize(image_rgb, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # ========================================
        # ASEGURAR TIPO DESPUÉS DE RESIZE
        # ========================================
        mask = mask.astype(np.int64)  # ← LÍNEA CRÍTICA AÑADIDA (por si resize cambió el tipo)

        # Aplicar transformaciones (augmentations)
        if self.transform:
            augmented = self.transform(image=image_rgb, mask=mask)
            image_rgb = augmented['image']
            mask = augmented['mask']

            # ========================================
            # FORZAR TIPO LONG DESPUÉS DE TRANSFORMS
            # ========================================
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).long()
            else:
                mask = mask.long()  # ← ASEGURAR QUE SEA LONG
        else:
            # Normalizar manualmente si no hay transform
            if self.normalize:
                image_rgb = image_rgb.astype(np.float32) / 255.0

            # Convertir a tensor CON TIPO CORRECTO
            image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()  # ← LONG, no byte

        return image_rgb, mask

    def get_image_name(self, idx: int) -> str:
        """Retorna nombre del archivo de imagen"""
        return self.image_files[idx]


def get_train_transform_grayscale(image_size: Tuple[int, int] = (512, 512), p: float = 0.7) -> A.Compose:
    """
    Transformaciones de entrenamiento para escala de grises
    Augmentation AGRESIVA optimizada para 35 imágenes

    Args:
        image_size: Tamaño de salida (altura, ancho)
        p: Probabilidad base de aplicar augmentations

    Returns:
        transform: Composición de transformaciones
    """
    return A.Compose([
        # ===== TRANSFORMACIONES GEOMÉTRICAS =====
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=p
        ),

        # ===== DISTORSIONES ÓPTICAS =====
        # Simulan variaciones microscópicas
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.15, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
        ], p=0.3),

        # ===== TRANSFORMACIONES DE INTENSIDAD =====
        # Apropiadas para fluorescencia
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=p
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),

        # ===== RUIDO =====
        # Simula variabilidad de cámara/microscopio
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.4),

        # ===== DESENFOQUE =====
        # Simula pérdida de foco
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.25),

        # ===== NORMALIZACIÓN =====
        # Usa estadísticas de ImageNet (necesario para encoder pre-entrenado)
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transform_grayscale(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """
    Transformaciones de validación (solo normalización, sin augmentation)

    Args:
        image_size: Tamaño de salida (altura, ancho)

    Returns:
        transform: Composición de transformaciones
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ===== TEST DEL DATASET =====
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("TEST DE DATASET EN ESCALA DE GRISES")
    print("=" * 60)

    # Crear dataset de ejemplo
    try:
        dataset = CometDatasetGrayscale(
            image_dir="dataset/images",
            mask_dir="dataset/masks",
            transform=get_train_transform_grayscale(),
            image_size=(512, 512)
        )

        print(f"\nTotal de imágenes: {len(dataset)}")

        if len(dataset) > 0:
            # Obtener primera imagen
            image, mask = dataset[0]

            print(f"\nImagenes cargadas correctamente:")
            print(f"  - Imagen shape: {image.shape}, dtype: {image.dtype}")
            print(f"  - Máscara shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"  - Valores únicos en máscara: {torch.unique(mask)}")

            # Verificar que los 3 canales son iguales (escala de grises)
            img_np = image.numpy()
            channels_equal = np.allclose(img_np[0], img_np[1], atol=1e-6) and np.allclose(img_np[1], img_np[2],
                                                                                          atol=1e-6)

            print(f"\n¿Los 3 canales RGB son idénticos? {channels_equal}")

            if channels_equal:
                print("✅ CORRECTO: La imagen está en escala de grises (3 canales idénticos)")
            else:
                print("⚠ ADVERTENCIA: Los canales RGB son diferentes (no es escala de grises pura)")

            # Visualizar
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Desnormalizar para visualización
            img_vis = img_np[0]  # Tomar solo el primer canal (son todos iguales)
            img_vis = img_vis * 0.229 + 0.485
            img_vis = np.clip(img_vis, 0, 1)

            # Imagen en escala de grises
            axes[0].imshow(img_vis, cmap='gray')
            axes[0].set_title("Imagen (Escala de Grises)")
            axes[0].axis('off')

            # Máscara con colores
            axes[1].imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=2)
            axes[1].set_title("Máscara\n(0=fondo, 1=cabeza, 2=cola)")
            axes[1].axis('off')

            # Overlay
            # Convertir máscara a color
            mask_color = np.zeros((*mask.shape, 3))
            mask_color[mask == 1] = [0, 1, 0]  # Verde para cabeza
            mask_color[mask == 2] = [1, 0, 0]  # Rojo para cola

            # Superponer
            overlay = np.stack([img_vis] * 3, axis=-1) * 0.7 + mask_color * 0.3
            overlay = np.clip(overlay, 0, 1)

            axes[2].imshow(overlay)
            axes[2].set_title("Overlay\n(Verde=Cabeza, Rojo=Cola)")
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig("dataset_grayscale_test.png", dpi=150, bbox_inches='tight')
            print(f"\n✅ Visualización guardada en: dataset_grayscale_test.png")
            print(f"\n{'=' * 60}")
            print("TEST COMPLETADO EXITOSAMENTE")
            print(f"{'=' * 60}")
        else:
            print("\n⚠ No hay imágenes en el dataset")
            print("Asegúrate de tener imágenes en dataset/images/")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nVerifica que:")
        print("  1. Existe la carpeta dataset/images/")
        print("  2. Existe la carpeta dataset/masks/")
        print("  3. Hay imágenes en dataset/images/")
        print("  4. Hay máscaras correspondientes en dataset/masks/")