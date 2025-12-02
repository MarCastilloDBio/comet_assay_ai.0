"""
Post-processing para separar cabeza/cola y calcular métricas
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


def preprocess_image(image: np.ndarray, apply_clahe: bool = True) -> np.ndarray:
    """
    Preprocesamiento de imagen: filtrado y realce de contraste

    Args:
        image: Imagen en escala de grises o RGB
        apply_clahe: Si True, aplica CLAHE

    Returns:
        processed: Imagen procesada
    """
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Filtro gaussiano para reducir ruido
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # Corrección de background (top-hat transform)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
    corrected = cv2.add(denoised, tophat)

    # CLAHE para realce de contraste
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        corrected = clahe.apply(corrected)

    return corrected


def clean_mask(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Limpia máscara eliminando regiones pequeñas

    Args:
        mask: Máscara binaria
        min_size: Tamaño mínimo de objeto en píxeles

    Returns:
        cleaned: Máscara limpia
    """
    # Convertir a binaria
    binary = mask > 0

    # Eliminar objetos pequeños
    cleaned = remove_small_objects(binary, min_size=min_size)

    # Operaciones morfológicas
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return cleaned.astype(np.uint8)


def separate_head_tail(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separa máscara en cabeza (clase 1) y cola (clase 2)

    Args:
        mask: Máscara multi-clase [H, W] con valores 0, 1, 2

    Returns:
        head_mask: Máscara binaria de cabeza
        tail_mask: Máscara binaria de cola
    """
    head_mask = (mask == 1).astype(np.uint8)
    tail_mask = (mask == 2).astype(np.uint8)

    # Limpiar máscaras
    head_mask = clean_mask(head_mask, min_size=50)
    tail_mask = clean_mask(tail_mask, min_size=50)

    return head_mask, tail_mask


def get_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Retorna el componente conectado más grande

    Args:
        mask: Máscara binaria

    Returns:
        largest: Máscara con solo el componente más grande
    """
    labeled = label(mask)

    if labeled.max() == 0:
        return mask

    # Encontrar región más grande
    regions = regionprops(labeled)
    largest_region = max(regions, key=lambda r: r.area)

    # Crear máscara con solo ese componente
    largest = (labeled == largest_region.label).astype(np.uint8)

    return largest


def calculate_skeleton_length(mask: np.ndarray) -> float:
    """
    Calcula longitud usando esqueletonización

    Args:
        mask: Máscara binaria

    Returns:
        length: Longitud en píxeles
    """
    if np.sum(mask) == 0:
        return 0.0

    # Esqueletonizar
    skeleton = skeletonize(mask > 0)

    # Contar píxeles del esqueleto
    length = np.sum(skeleton)

    return float(length)


def calculate_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """
    Calcula centroide de una máscara

    Args:
        mask: Máscara binaria

    Returns:
        cy, cx: Coordenadas del centroide
    """
    if np.sum(mask) == 0:
        return 0.0, 0.0

    # Usar ndimage para centroide
    cy, cx = ndimage.center_of_mass(mask)

    return cy, cx


def calculate_euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calcula distancia euclidiana entre dos puntos"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_metrics(
        image: np.ndarray,
        head_mask: np.ndarray,
        tail_mask: np.ndarray,
        pixel_size_um: float = None
) -> Dict[str, float]:
    """
    Calcula todas las métricas para un cometa

    Args:
        image: Imagen original en escala de grises
        head_mask: Máscara binaria de cabeza
        tail_mask: Máscara binaria de cola
        pixel_size_um: Tamaño de píxel en micrómetros (opcional)

    Returns:
        metrics: Diccionario con todas las métricas
    """
    # Obtener componentes más grandes
    head_mask = get_largest_component(head_mask)
    tail_mask = get_largest_component(tail_mask)

    # Máscara total del cometa
    comet_mask = ((head_mask + tail_mask) > 0).astype(np.uint8)

    # --- INTENSIDADES ---
    # Calcular intensidades totales
    head_intensity_total = np.sum(image[head_mask > 0]) if np.any(head_mask) else 0
    tail_intensity_total = np.sum(image[tail_mask > 0]) if np.any(tail_mask) else 0
    total_intensity = head_intensity_total + tail_intensity_total

    # Intensidades medias
    head_intensity_mean = np.mean(image[head_mask > 0]) if np.any(head_mask) else 0
    tail_intensity_mean = np.mean(image[tail_mask > 0]) if np.any(tail_mask) else 0

    # --- ÁREAS ---
    head_area_px = np.sum(head_mask)
    tail_area_px = np.sum(tail_mask)
    total_area_px = np.sum(comet_mask)

    # --- TAIL DNA % ---
    tail_dna_percent = (tail_intensity_total / total_intensity * 100) if total_intensity > 0 else 0

    # --- CENTROIDES Y DISTANCIA ---
    head_cy, head_cx = calculate_centroid(head_mask)
    tail_cy, tail_cx = calculate_centroid(tail_mask)

    distance_centroids_px = calculate_euclidean_distance(
        (head_cy, head_cx),
        (tail_cy, tail_cx)
    )

    # --- TAIL MOMENT ---
    # Tail Moment = Tail DNA % × distancia entre centroides
    tail_moment = tail_dna_percent * distance_centroids_px

    # --- LONGITUDES ---
    # Longitud total del cometa (esqueletonización)
    comet_length_px = calculate_skeleton_length(comet_mask)

    # Longitudes de cabeza y cola
    head_length_px = calculate_skeleton_length(head_mask)
    tail_length_px = calculate_skeleton_length(tail_mask)

    # --- CONVERSIÓN A MICRÓMETROS (si disponible) ---
    if pixel_size_um:
        distance_centroids_um = distance_centroids_px * pixel_size_um
        comet_length_um = comet_length_px * pixel_size_um
        head_length_um = head_length_px * pixel_size_um
        tail_length_um = tail_length_px * pixel_size_um
        head_area_um2 = head_area_px * (pixel_size_um ** 2)
        tail_area_um2 = tail_area_px * (pixel_size_um ** 2)
        total_area_um2 = total_area_px * (pixel_size_um ** 2)
    else:
        distance_centroids_um = None
        comet_length_um = None
        head_length_um = None
        tail_length_um = None
        head_area_um2 = None
        tail_area_um2 = None
        total_area_um2 = None

    # --- RATIO TAIL/HEAD ---
    tail_head_ratio = tail_intensity_total / head_intensity_total if head_intensity_total > 0 else 0

    # Compilar métricas
    metrics = {
        # Intensidades
        'head_intensity_total': float(head_intensity_total),
        'tail_intensity_total': float(tail_intensity_total),
        'total_intensity': float(total_intensity),
        'head_intensity_mean': float(head_intensity_mean),
        'tail_intensity_mean': float(tail_intensity_mean),

        # Áreas (píxeles)
        'head_area_px': int(head_area_px),
        'tail_area_px': int(tail_area_px),
        'total_area_px': int(total_area_px),

        # Métricas biológicas principales
        'tail_dna_percent': float(tail_dna_percent),
        'tail_moment': float(tail_moment),
        'tail_head_ratio': float(tail_head_ratio),

        # Distancias y longitudes (píxeles)
        'distance_centroids_px': float(distance_centroids_px),
        'comet_length_px': float(comet_length_px),
        'head_length_px': float(head_length_px),
        'tail_length_px': float(tail_length_px),

        # Centroides
        'head_centroid_y': float(head_cy),
        'head_centroid_x': float(head_cx),
        'tail_centroid_y': float(tail_cy),
        'tail_centroid_x': float(tail_cx),
    }

    # Agregar métricas en micrómetros si disponible
    if pixel_size_um:
        metrics.update({
            'distance_centroids_um': float(distance_centroids_um),
            'comet_length_um': float(comet_length_um),
            'head_length_um': float(head_length_um),
            'tail_length_um': float(tail_length_um),
            'head_area_um2': float(head_area_um2),
            'tail_area_um2': float(tail_area_um2),
            'total_area_um2': float(total_area_um2),
            'pixel_size_um': float(pixel_size_um),
        })

    return metrics


def classify_damage_level(tail_dna_percent: float) -> str:
    """
    Clasifica el nivel de daño según Tail DNA %

    Args:
        tail_dna_percent: Porcentaje de ADN en cola

    Returns:
        level: Nivel de daño ('minimal', 'low', 'moderate', 'severe')
    """
    if tail_dna_percent < 5:
        return 'minimal'
    elif tail_dna_percent < 20:
        return 'low'
    elif tail_dna_percent < 40:
        return 'moderate'
    else:
        return 'severe'


def visualize_segmentation(
        image: np.ndarray,
        mask: np.ndarray,
        head_mask: np.ndarray,
        tail_mask: np.ndarray,
        alpha: float = 0.5
) -> np.ndarray:
    """
    Crea visualización con overlay de máscaras

    Args:
        image: Imagen original (escala de grises)
        mask: Máscara multi-clase
        head_mask: Máscara de cabeza
        tail_mask: Máscara de cola
        alpha: Transparencia del overlay

    Returns:
        overlay: Imagen RGB con máscaras superpuestas
    """
    # Convertir imagen a RGB
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    # Normalizar a rango [0, 255]
    vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Crear overlay de colores
    overlay = vis.copy()

    # Cabeza en verde
    overlay[head_mask > 0] = [0, 255, 0]

    # Cola en rojo
    overlay[tail_mask > 0] = [255, 0, 0]

    # Mezclar con imagen original
    result = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

    # Dibujar contornos
    head_contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tail_contours, _ = cv2.findContours(tail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(result, head_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(result, tail_contours, -1, (255, 0, 0), 2)

    # Marcar centroides
    head_cy, head_cx = calculate_centroid(head_mask)
    tail_cy, tail_cx = calculate_centroid(tail_mask)

    if head_cy > 0 and head_cx > 0:
        cv2.circle(result, (int(head_cx), int(head_cy)), 5, (0, 255, 0), -1)

    if tail_cy > 0 and tail_cx > 0:
        cv2.circle(result, (int(tail_cx), int(tail_cy)), 5, (255, 0, 0), -1)

    # Dibujar línea entre centroides
    if head_cy > 0 and tail_cy > 0:
        cv2.line(
            result,
            (int(head_cx), int(head_cy)),
            (int(tail_cx), int(tail_cy)),
            (255, 255, 0),
            2
        )

    return result


# Test de post-processing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Crear máscaras sintéticas para prueba
    print("Generando máscaras sintéticas...")

    # Imagen de prueba
    image = np.random.randint(50, 200, (512, 512), dtype=np.uint8)

    # Máscara sintética
    mask = np.zeros((512, 512), dtype=np.uint8)

    # Cabeza (círculo)
    cv2.circle(mask, (256, 200), 40, 1, -1)

    # Cola (elipse alargada)
    cv2.ellipse(mask, (256, 300), (30, 80), 0, 0, 360, 2, -1)

    # Separar y calcular métricas
    head_mask, tail_mask = separate_head_tail(mask)
    metrics = calculate_metrics(image, head_mask, tail_mask, pixel_size_um=0.65)

    # Visualizar
    overlay = visualize_segmentation(image, mask, head_mask, tail_mask)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Imagen Original")
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='tab10', vmin=0, vmax=2)
    axes[1].set_title("Máscara Multi-clase")
    axes[1].axis('off')

    axes[2].imshow(head_mask, cmap='Greens')
    axes[2].set_title("Cabeza")
    axes[2].axis('off')

    axes[3].imshow(tail_mask, cmap='Reds')
    axes[3].set_title("Cola")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig("postprocessing_test.png", dpi=150)
    print("Test guardado en postprocessing_test.png")

    # Imprimir métricas
    print("\nMétricas calculadas:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.3f}")
        else:
            print(f"{key:30s}: {value}")

    print(f"\nNivel de daño: {classify_damage_level(metrics['tail_dna_percent'])}")