import math
import numpy as np
import random
from abc import ABC , abstractmethod
import torch

# ==========================================
# Anstract Class for Density
# ==========================================

class Density(ABC) :
    "Abstract Class for density"
    
    @abstractmethod
    def energy(self,x) :
        "Return the Energy U(x) "
        pass

    @abstractmethod
    def sample(self, num_samples, device) :
        """ Sampling directly from the distribution (without constraint) """
        pass


# ==========================================
# Anstract Class for Density
# ==========================================

class Gaussienne(Density) : 
    def __init__(self, mean, cov) :
        self.mean = torch.tensor(mean, dtype=torch.float32)

        dim = self.mean.numel()
        self.cov = torch.tensor(cov, dtype=torch.float32)
        self.inv_cov = torch.linalg.inv(self.cov)
        self.chol = torch.linalg.cholesky(self.cov)

    def energy(self,x) :
        device = x.device 
        mean = self.mean.to(device)
        cov = self.cov.to(device)

        # Inverse of covariance 
        inv_cov = torch.linalg.inv(cov)

        #diff
        diff = x - mean

        #left term
        left_term = diff @ inv_cov

        return 0.5* torch.sum(left_term * diff, dim = 1)
    
    def sample (self,num_samples,device) :
        
        mean = self.mean.to(device)
        # We use Cholesky decomposition
        chol = self.chol.to(device=device)
        
        # Genereted a gaussian white noise
        eps = torch.randn((num_samples, len(mean)), device=device)
        
        # retirn sample
        return mean + eps @ chol.T
