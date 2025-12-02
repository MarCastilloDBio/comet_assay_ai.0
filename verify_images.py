"""
Verificar que las 35 imágenes son válidas y analizarlas
"""

import cv2
import numpy as np
from pathlib import Path


def verify_images():
    image_dir = Path("dataset/images")

    images = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))

    print(f"Total de imágenes encontradas: {len(images)}\n")
    print("=" * 80)

    issues = []
    color_stats = {'ROJO': 0, 'VERDE': 0, 'AZUL': 0, 'GRIS': 0}

    for img_path in images:
        img = cv2.imread(str(img_path))

        if img is None:
            issues.append(f"❌ {img_path.name}: No se puede leer")
            continue

        h, w = img.shape[:2]

        # Analizar color predominante
        if len(img.shape) == 3:
            b, g, r = cv2.split(img)
            b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()

            if r_mean > g_mean and r_mean > b_mean:
                color = "ROJO"
            elif g_mean > r_mean and g_mean > b_mean:
                color = "VERDE"
            elif b_mean > r_mean and b_mean > g_mean:
                color = "AZUL"
            else:
                color = "GRIS"

            color_stats[color] += 1
        else:
            color = "GRIS"
            color_stats[color] += 1

        # Verificaciones
        status = "✓"
        if w < 100 or h < 100:
            status = "⚠"
            issues.append(f"⚠ {img_path.name}: Muy pequeña ({w}x{h})")

        print(f"{status} {img_path.name:20s} | {w:4d}x{h:4d} | Color: {color:6s}")

    print("=" * 80)
    print(f"\nRESUMEN DE COLORES:")
    for color, count in color_stats.items():
        if count > 0:
            print(f"  {color:6s}: {count:2d} imágenes ({count / len(images) * 100:.1f}%)")

    print(
        f"\n{'✓ DATASET MIXTO - Perfecto para escala de grises' if len([c for c in color_stats.values() if c > 0]) > 1 else '⚠ Dataset homogéneo'}")

    if issues:
        print(f"\n⚠ PROBLEMAS DETECTADOS:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n✓ Todas las imágenes son válidas")

    print(f"\n✓ Total: {len(images)} imágenes listas para etiquetar")


if __name__ == "__main__":
    verify_images()