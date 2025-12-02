"""
Funciones de métricas para evaluación de segmentación
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """
    Calcula Dice coefficient (F1-score para segmentación)

    Args:
        pred: Máscara predicha binaria
        target: Máscara ground truth binaria
        smooth: Factor de suavizado

    Returns:
        dice: Coeficiente Dice [0, 1]
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    intersection = np.sum(pred & target)
    union = np.sum(pred) + np.sum(target)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return float(dice)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calcula Intersection over Union (IoU / Jaccard Index)

    Args:
        pred: Máscara predicha binaria
        target: Máscara ground truth binaria
        smooth: Factor de suavizado

    Returns:
        iou: IoU score [0, 1]
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    intersection = np.sum(pred & target)
    union = np.sum(pred | target)

    iou = (intersection + smooth) / (union + smooth)

    return float(iou)


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calcula precisión a nivel de píxel

    Args:
        pred: Máscara predicha
        target: Máscara ground truth

    Returns:
        accuracy: Precisión [0, 1]
    """
    correct = np.sum(pred == target)
    total = pred.size

    return float(correct / total)


def precision_recall(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Calcula precision y recall

    Args:
        pred: Máscara predicha binaria
        target: Máscara ground truth binaria

    Returns:
        metrics: Dict con precision y recall
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    tp = np.sum(pred & target)
    fp = np.sum(pred & ~target)
    fn = np.sum(~pred & target)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def calculate_multiclass_metrics(
        pred: np.ndarray,
        target: np.ndarray,
        num_classes: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    Calcula métricas por clase para segmentación multi-clase

    Args:
        pred: Máscara predicha [H, W] con valores 0, 1, 2
        target: Máscara ground truth [H, W] con valores 0, 1, 2
        num_classes: Número de clases

    Returns:
        metrics: Dict con métricas por clase
    """
    class_names = ['background', 'head', 'tail']
    metrics = {}

    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        dice = dice_score(pred_c, target_c)
        iou = iou_score(pred_c, target_c)
        pr = precision_recall(pred_c, target_c)

        class_name = class_names[c] if c < len(class_names) else f'class_{c}'

        metrics[class_name] = {
            'dice': dice,
            'iou': iou,
            'precision': pr['precision'],
            'recall': pr['recall'],
            'f1': pr['f1']
        }

    # Calcular promedio (excluyendo background)
    if num_classes > 1:
        avg_dice = np.mean([metrics[class_names[i]]['dice'] for i in range(1, min(num_classes, len(class_names)))])
        avg_iou = np.mean([metrics[class_names[i]]['iou'] for i in range(1, min(num_classes, len(class_names)))])

        metrics['mean'] = {
            'dice': float(avg_dice),
            'iou': float(avg_iou)
        }

    return metrics


def confusion_matrix_multiclass(
        pred: np.ndarray,
        target: np.ndarray,
        num_classes: int = 3
) -> np.ndarray:
    """
    Calcula matriz de confusión

    Args:
        pred: Máscara predicha
        target: Máscara ground truth
        num_classes: Número de clases

    Returns:
        cm: Matriz de confusión [num_classes, num_classes]
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    cm = confusion_matrix(target_flat, pred_flat, labels=range(num_classes))

    return cm


def mean_absolute_error(pred_values: List[float], target_values: List[float]) -> float:
    """
    Calcula MAE entre valores predichos y reales

    Args:
        pred_values: Lista de valores predichos
        target_values: Lista de valores ground truth

    Returns:
        mae: Error absoluto medio
    """
    pred_arr = np.array(pred_values)
    target_arr = np.array(target_values)

    mae = np.mean(np.abs(pred_arr - target_arr))

    return float(mae)


def correlation_coefficient(pred_values: List[float], target_values: List[float]) -> float:
    """
    Calcula coeficiente de correlación de Pearson

    Args:
        pred_values: Lista de valores predichos
        target_values: Lista de valores ground truth

    Returns:
        r: Coeficiente de correlación [-1, 1]
    """
    pred_arr = np.array(pred_values)
    target_arr = np.array(target_values)

    if len(pred_arr) < 2:
        return 0.0

    r = np.corrcoef(pred_arr, target_arr)[0, 1]

    return float(r) if not np.isnan(r) else 0.0


def bland_altman_statistics(
        pred_values: List[float],
        target_values: List[float]
) -> Dict[str, float]:
    """
    Calcula estadísticas de Bland-Altman para concordancia

    Args:
        pred_values: Lista de valores predichos
        target_values: Lista de valores ground truth

    Returns:
        stats: Dict con bias, límites de concordancia, etc.
    """
    pred_arr = np.array(pred_values)
    target_arr = np.array(target_values)

    # Diferencias
    diff = pred_arr - target_arr

    # Media de diferencias (bias)
    bias = np.mean(diff)

    # Desviación estándar de diferencias
    std_diff = np.std(diff, ddof=1)

    # Límites de concordancia (±1.96 SD)
    upper_loa = bias + 1.96 * std_diff
    lower_loa = bias - 1.96 * std_diff

    return {
        'bias': float(bias),
        'std_diff': float(std_diff),
        'upper_loa': float(upper_loa),
        'lower_loa': float(lower_loa)
    }


# Test de métricas
if __name__ == "__main__":
    print("Test de métricas\n")

    # Máscaras sintéticas
    pred = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 2],
        [1, 1, 2, 2],
        [1, 2, 2, 2]
    ])

    target = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 2, 2],
        [1, 2, 2, 2]
    ])

    # Métricas multi-clase
    metrics = calculate_multiclass_metrics(pred, target, num_classes=3)

    print("Métricas por clase:")
    for class_name, class_metrics in metrics.items():
        print(f"\n{class_name.upper()}:")
        for metric_name, value in class_metrics.items():
            print(f"  {metric_name:12s}: {value:.4f}")

    # Matriz de confusión
    cm = confusion_matrix_multiclass(pred, target, num_classes=3)
    print("\nMatriz de Confusión:")
    print(cm)

    # Test de métricas continuas
    pred_values = [10.5, 20.3, 15.7, 30.1, 25.8]
    target_values = [11.0, 19.8, 16.2, 29.5, 26.0]

    mae = mean_absolute_error(pred_values, target_values)
    r = correlation_coefficient(pred_values, target_values)
    ba_stats = bland_altman_statistics(pred_values, target_values)

    print(f"\nMétricas continuas:")
    print(f"MAE:         {mae:.4f}")
    print(f"Correlación: {r:.4f}")
    print(f"Bias:        {ba_stats['bias']:.4f}")
    print(f"Upper LoA:   {ba_stats['upper_loa']:.4f}")
    print(f"Lower LoA:   {ba_stats['lower_loa']:.4f}")