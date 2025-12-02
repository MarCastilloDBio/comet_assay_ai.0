"""
Script para convertir anotaciones de LabelMe a máscaras PNG
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd


def labelme_to_mask(
        json_path: str,
        image_shape: tuple,
        label_mapping: dict = {'head': 1, 'tail': 2}
) -> np.ndarray:
    """
    Convierte anotación LabelMe JSON a máscara multi-clase

    Args:
        json_path: Path al archivo JSON de LabelMe
        image_shape: (height, width) de la imagen
        label_mapping: Mapeo de labels a valores de clase

    Returns:
        mask: Máscara [H, W] con valores 0 (fondo), 1 (cabeza), 2 (cola)
    """
    # Cargar JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Crear máscara vacía
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Procesar cada shape
    for shape in data.get('shapes', []):
        label = shape['label'].lower()
        points = shape['points']
        shape_type = shape['shape_type']

        # Determinar valor de clase
        class_value = label_mapping.get(label, 0)

        if class_value == 0:
            continue  # Skip fondo o labels desconocidos

        # Convertir puntos a array
        points_array = np.array(points, dtype=np.int32)

        # Dibujar según tipo de shape
        if shape_type == 'polygon':
            cv2.fillPoly(mask, [points_array], class_value)
        elif shape_type == 'rectangle':
            x1, y1 = points_array[0]
            x2, y2 = points_array[1]
            cv2.rectangle(mask, (x1, y1), (x2, y2), class_value, -1)
        elif shape_type == 'circle':
            center = tuple(points_array[0])
            edge = tuple(points_array[1])
            radius = int(np.linalg.norm(points_array[1] - points_array[0]))
            cv2.circle(mask, center, radius, class_value, -1)

    return mask


def process_labelme_annotations(
        annotations_dir: str,
        images_dir: str,
        output_masks_dir: str,
        label_mapping: dict = {'head': 1, 'tail': 2}
) -> pd.DataFrame:
    """
    Procesa todas las anotaciones LabelMe y genera máscaras

    Args:
        annotations_dir: Directorio con archivos JSON
        images_dir: Directorio con imágenes originales
        output_masks_dir: Directorio de salida para máscaras
        label_mapping: Mapeo de labels a clases

    Returns:
        df: DataFrame con información de máscaras generadas
    """
    annotations_dir = Path(annotations_dir)
    images_dir = Path(images_dir)
    output_masks_dir = Path(output_masks_dir)

    output_masks_dir.mkdir(parents=True, exist_ok=True)

    # Buscar archivos JSON
    json_files = list(annotations_dir.glob('*.json'))
    print(f"Encontrados {len(json_files)} archivos JSON")

    results = []

    for json_path in tqdm(json_files, desc="Convirtiendo anotaciones"):
        try:
            # Cargar JSON para obtener nombre de imagen
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_filename = data.get('imagePath', json_path.stem + '.png')
            image_path = images_dir / image_filename

            # Cargar imagen para obtener dimensiones
            if not image_path.exists():
                # Intentar con extensiones alternativas
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    alt_path = images_dir / (json_path.stem + ext)
                    if alt_path.exists():
                        image_path = alt_path
                        break

            if not image_path.exists():
                print(f"Advertencia: Imagen no encontrada para {json_path.name}")
                continue

            # Leer imagen para obtener dimensiones
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Advertencia: No se pudo leer {image_path}")
                continue

            height, width = img.shape[:2]

            # Generar máscara
            mask = labelme_to_mask(
                str(json_path),
                image_shape=(height, width),
                label_mapping=label_mapping
            )

            # Guardar máscara
            mask_filename = json_path.stem + '.png'
            mask_path = output_masks_dir / mask_filename
            cv2.imwrite(str(mask_path), mask)

            # Calcular estadísticas
            head_pixels = np.sum(mask == 1)
            tail_pixels = np.sum(mask == 2)
            total_pixels = head_pixels + tail_pixels

            # Calcular bounding box
            if total_pixels > 0:
                coords = np.argwhere(mask > 0)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                bbox = f"{x_min},{y_min},{x_max},{y_max}"
            else:
                bbox = "0,0,0,0"

            results.append({
                'image_name': image_filename,
                'mask_name': mask_filename,
                'image_path': str(image_path),
                'mask_path': str(mask_path),
                'width': width,
                'height': height,
                'head_pixels': int(head_pixels),
                'tail_pixels': int(tail_pixels),
                'total_pixels': int(total_pixels),
                'bbox': bbox
            })

        except Exception as e:
            print(f"Error procesando {json_path.name}: {e}")
            continue

    # Crear DataFrame
    df = pd.DataFrame(results)

    return df


def visualize_mask(image_path: str, mask_path: str, output_path: str):
    """Crea visualización con overlay de máscara"""
    # Cargar imagen y máscara
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        return

    # Convertir a RGB si es necesario
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Crear overlay
    overlay = image.copy()
    overlay[mask == 1] = [0, 255, 0]  # Cabeza en verde
    overlay[mask == 2] = [255, 0, 0]  # Cola en rojo

    # Mezclar
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    # Guardar
    cv2.imwrite(output_path, result)


def main():
    parser = argparse.ArgumentParser(
        description='Convertir anotaciones LabelMe a máscaras PNG'
    )

    parser.add_argument(
        '--annotations',
        type=str,
        required=True,
        help='Directorio con archivos JSON de LabelMe'
    )
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Directorio con imágenes originales'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Directorio de salida para máscaras'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generar visualizaciones con overlay'
    )

    args = parser.parse_args()

    # Label mapping
    label_mapping = {
        'head': 1,
        'cabeza': 1,
        'tail': 2,
        'cola': 2
    }

    print("Convirtiendo anotaciones LabelMe a máscaras...\n")

    # Procesar anotaciones
    df = process_labelme_annotations(
        annotations_dir=args.annotations,
        images_dir=args.images,
        output_masks_dir=args.output,
        label_mapping=label_mapping
    )

    # Guardar CSV con información
    csv_path = Path(args.output).parent / 'masks_info.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Información guardada en: {csv_path}")

    # Estadísticas
    print("\nEstadísticas:")
    print(f"Total de máscaras generadas: {len(df)}")
    print(f"Píxeles promedio cabeza: {df['head_pixels'].mean():.0f}")
    print(f"Píxeles promedio cola: {df['tail_pixels'].mean():.0f}")

    # Generar visualizaciones si se solicita
    if args.visualize:
        print("\nGenerando visualizaciones...")
        vis_dir = Path(args.output).parent / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Visualizando"):
            vis_path = vis_dir / row['mask_name']
            visualize_mask(row['image_path'], row['mask_path'], str(vis_path))

        print(f"✓ Visualizaciones guardadas en: {vis_dir}")

    print(f"\n✓ Proceso completado. Máscaras en: {args.output}")


if __name__ == "__main__":
    main()
    