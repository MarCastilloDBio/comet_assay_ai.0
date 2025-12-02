"""
Script de inferencia para procesar imágenes con modelo entrenado
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import json

import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd

from model import UNetComet
from postprocessing import (
    preprocess_image,
    separate_head_tail,
    calculate_metrics,
    visualize_segmentation
)
from utils import load_model


class CometInference:
    """
    Clase para realizar inferencia en imágenes de cometa
    """

    def __init__(
            self,
            model_path: str,
            device: str = 'cuda',
            image_size: int = 512,
            pixel_size_um: float = None
    ):
        """
        Args:
            model_path: Path al modelo entrenado (.pth)
            device: 'cuda' o 'cpu'
            image_size: Tamaño de entrada del modelo
            pixel_size_um: Tamaño de píxel en micrómetros (para conversión física)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.pixel_size_um = pixel_size_um

        # Cargar modelo
        print(f"Cargando modelo desde: {model_path}")
        self.model = load_model(model_path, device=self.device)
        self.model.eval()

        print(f"Modelo cargado en: {self.device}")

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocesa imagen para inferencia

        Args:
            image: Imagen numpy [H, W] o [H, W, C]

        Returns:
            tensor: Tensor [1, 3, H, W] listo para modelo
            original: Imagen original procesada
        """
        # Aplicar preprocesamiento
        processed = preprocess_image(image)

        # FORZAR conversión a escala de grises primero (consistencia)
        if len(processed.shape) == 3:
            # Si es color, convertir a grises
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Ahora convertir a 3 canales RGB (todos con mismo valor)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

        # Redimensionar
        h, w = processed.shape[:2]
        resized = cv2.resize(processed, (self.image_size, self.image_size))

        # Normalizar (ImageNet stats)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # Convertir a tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()

        return tensor, processed

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predice máscara para una imagen

        Args:
            image: Imagen numpy

        Returns:
            mask: Máscara predicha [H, W] con valores 0, 1, 2
        """
        # Preprocesar
        tensor, original = self.preprocess(image)
        tensor = tensor.to(self.device)

        # Inferencia
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

        # Redimensionar a tamaño original
        h, w = image.shape[:2]
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        return mask

    def process_image(
            self,
            image_path: str,
            output_dir: str = None,
            save_overlay: bool = True
    ) -> dict:
        """
        Procesa una imagen completa: inferencia + post-processing + métricas

        Args:
            image_path: Path a la imagen
            output_dir: Directorio para guardar resultados
            save_overlay: Si True, guarda visualización con overlay

        Returns:
            results: Diccionario con métricas
        """
        # Cargar imagen
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"No se pudo cargar: {image_path}")

        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Predecir máscara
        mask = self.predict(image)

        # Separar cabeza y cola
        head_mask, tail_mask = separate_head_tail(mask)

        # Calcular métricas
        metrics = calculate_metrics(
            image_gray,
            head_mask,
            tail_mask,
            pixel_size_um=self.pixel_size_um
        )

        # Agregar metadatos
        metrics['image_name'] = Path(image_path).name
        metrics['image_path'] = image_path

        # Guardar overlay si se solicita
        if output_dir and save_overlay:
            output_path = Path(output_dir) / 'overlays' / Path(image_path).name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            overlay = visualize_segmentation(image_gray, mask, head_mask, tail_mask)
            cv2.imwrite(str(output_path), overlay)

            metrics['overlay_path'] = str(output_path)

        return metrics

    def process_batch(
            self,
            image_paths: List[str],
            output_dir: str,
            save_csv: bool = True
    ) -> pd.DataFrame:
        """
        Procesa un lote de imágenes

        Args:
            image_paths: Lista de paths a imágenes
            output_dir: Directorio de salida
            save_csv: Si True, guarda CSV con resultados

        Returns:
            df: DataFrame con todas las métricas
        """
        results = []

        for img_path in tqdm(image_paths, desc="Procesando imágenes"):
            try:
                metrics = self.process_image(
                    img_path,
                    output_dir=output_dir,
                    save_overlay=True
                )
                results.append(metrics)
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                continue

        # Convertir a DataFrame
        df = pd.DataFrame(results)

        # Guardar CSV
        if save_csv:
            csv_path = Path(output_dir) / 'metrics.csv'
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Métricas guardadas en: {csv_path}")

        return df


def main():
    parser = argparse.ArgumentParser(description='Inferencia en imágenes de cometa')

    parser.add_argument('--model', type=str, required=True, help='Path al modelo (.pth)')
    parser.add_argument('--image', type=str, default=None, help='Path a imagen individual')
    parser.add_argument('--image_dir', type=str, default=None, help='Directorio con imágenes')
    parser.add_argument('--output', type=str, default='results', help='Directorio de salida')
    parser.add_argument('--image_size', type=int, default=512, help='Tamaño de entrada')
    parser.add_argument('--pixel_size_um', type=float, default=None,
                        help='Tamaño de píxel en micrómetros')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Validar argumentos
    if args.image is None and args.image_dir is None:
        raise ValueError("Debe proporcionar --image o --image_dir")

    # Crear inferencia
    inference = CometInference(
        model_path=args.model,
        device=args.device,
        image_size=args.image_size,
        pixel_size_um=args.pixel_size_um
    )

    # Crear directorio de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Procesar imagen(es)
    if args.image:
        # Imagen individual
        print(f"\nProcesando imagen: {args.image}")
        metrics = inference.process_image(
            args.image,
            output_dir=args.output,
            save_overlay=True
        )

        # Imprimir resultados
        print("\n" + "=" * 60)
        print("RESULTADOS")
        print("=" * 60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:.3f}")
            else:
                print(f"{key:30s}: {value}")

        # Guardar JSON
        json_path = output_dir / f"{Path(args.image).stem}_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Métricas guardadas en: {json_path}")

    else:
        # Directorio de imágenes
        print(f"\nBuscando imágenes en: {args.image_dir}")
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            image_paths.extend(Path(args.image_dir).glob(ext))

        print(f"Encontradas {len(image_paths)} imágenes")

        if len(image_paths) == 0:
            print("No se encontraron imágenes")
            return

        # Procesar batch
        df = inference.process_batch(
            [str(p) for p in image_paths],
            output_dir=args.output,
            save_csv=True
        )

        # Estadísticas
        print("\n" + "=" * 60)
        print("ESTADÍSTICAS")
        print("=" * 60)
        print(df[['tail_dna_percent', 'tail_moment', 'comet_length_px']].describe())

        print(f"\n✓ Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()