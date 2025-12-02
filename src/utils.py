"""
Utilidades generales para el proyecto
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def load_model(model_path: str, device: torch.device = None) -> torch.nn.Module:
    """
    Carga modelo desde checkpoint

    Args:
        model_path: Path al checkpoint (.pth)
        device: Device donde cargar el modelo

    Returns:
        model: Modelo cargado y en modo eval
    """
    from model import UNetComet

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Crear modelo
    model = UNetComet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=3,
        classes=3
    )

    # Cargar pesos
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def save_config(config: Dict[str, Any], output_path: str):
    """Guarda configuración en JSON o YAML"""
    output_path = Path(output_path)

    if output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    elif output_path.suffix in ['.yaml', '.yml']:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Formato no soportado: {output_path.suffix}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Carga configuración desde JSON o YAML"""
    config_path = Path(config_path)

    if config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Formato no soportado: {config_path.suffix}")

    return config


def convert_px_to_um(
        length_px: float,
        pixel_size_um: float,
        magnification: Optional[int] = None
) -> float:
    """
    Convierte longitud de píxeles a micrómetros

    Args:
        length_px: Longitud en píxeles
        pixel_size_um: Tamaño de píxel en µm
        magnification: Magnificación del objetivo (opcional)

    Returns:
        length_um: Longitud en micrómetros
    """
    length_um = length_px * pixel_size_um

    if magnification:
        length_um = length_um / magnification

    return length_um


def extract_pixel_size_from_tiff(tiff_path: str) -> Optional[float]:
    """
    Extrae tamaño de píxel desde metadatos TIFF

    Args:
        tiff_path: Path a archivo TIFF

    Returns:
        pixel_size_um: Tamaño de píxel en µm o None
    """
    try:
        from PIL import Image
        from PIL.TiffTags import TAGS

        img = Image.open(tiff_path)

        # Intentar obtener resolución
        if 'dpi' in img.info:
            dpi = img.info['dpi'][0]
            # Convertir DPI a µm/pixel
            pixel_size_um = 25400.0 / dpi
            return pixel_size_um

        # Buscar en tags TIFF
        for tag_id, value in img.tag_v2.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if 'Resolution' in str(tag_name):
                # Procesamiento específico según formato
                pass

        return None

    except Exception as e:
        print(f"Error extrayendo metadatos: {e}")
        return None


def ensure_dir(path: str):
    """Crea directorio si no existe"""
    Path(path).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """Cuenta parámetros entrenables del modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42):
    """Establece seed para reproducibilidad"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Calcula y almacena promedio y valor actual"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Test de utilidades
if __name__ == "__main__":
    print("Test de utilidades\n")

    # Test conversión px → µm
    length_px = 150
    pixel_size = 0.65
    length_um = convert_px_to_um(length_px, pixel_size)
    print(f"Conversión: {length_px} px = {length_um:.2f} µm (pixel_size={pixel_size} µm)")

    # Test AverageMeter
    meter = AverageMeter()
    for val in [1.5, 2.0, 1.8, 2.2]:
        meter.update(val)
    print(f"\nAverageMeter: avg={meter.avg:.3f}, count={meter.count}")

    # Test seed
    set_seed(42)
    print(f"\nSeed establecida: {torch.initial_seed()}")

    print("\n✓ Test completado")