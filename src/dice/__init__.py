"""
DICE - Discrete Inter-model Comparison Evaluator
面向检索增强型生成（RAG）系统的离散化评估框架
"""

from .dice_core import DICEEvaluator, DICEConfig, create_dice_evaluator
from .granularity import GranularityBuilder
from .local_pairwise_judge import LocalPairwiseJudge as PairwiseJudge
from .elo_fusion import EloFusion

__version__ = "1.0.0"
__all__ = ["DICEEvaluator", "DICEConfig", "create_dice_evaluator", "GranularityBuilder", "PairwiseJudge", "EloFusion"] 