"""
Comet Assay AI - Automated Genotoxicity Analysis
Versión optimizada para escala de grises
"""

__version__ = "1.0.0"
__author__ = "Comet Assay Team"

from .model import UNetComet
from .dataset_grayscale import CometDatasetGrayscale
from .postprocessing import separate_head_tail, calculate_metrics
from .metrics import dice_score, iou_score

__all__ = [
    'UNetComet',
    'CometDatasetGrayscale',
    'separate_head_tail',
    'calculate_metrics',
    'dice_score',
    'iou_score',
]