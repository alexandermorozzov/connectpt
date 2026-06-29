"""Training application: data, trainers and runs for route models.

Depends on core/ models/ objectives/ and the existing improvement_learning
trainer. Must NOT import the search layer (the two are separate applications
connected only through checkpoints + configs).
"""

from .training_data import TrainingDataModule


__all__ = ["TrainingDataModule"]
