

import math
import numpy as np
import random
from abc import ABC , abstractmethod
import torch

#===========================================
# Anstract Class for Constraint
#==========================================

class Constraint(ABC):
    """Abstract Class"""
    
    @abstractmethod
    def evaluate(self, x):
        pass

class EllipsoidConstraint(Constraint):
    """Elipsoid Constraint"""
    
    def __init__(self, radius_x, radius_y, center=(0.0, 0.0)):
        self.rx = radius_x
        self.ry = radius_y
        self.center = torch.tensor(center, dtype=torch.float32)

    def evaluate(self, x):
        device = x.device
        c = self.center.to(device)
        return ((x[:, 0] - c[0]) / self.rx)**2 + ((x[:, 1] - c[1]) / self.ry)**2 - 1.0