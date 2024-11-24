from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict
import torch

class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        """
        Push a new value to the running window
        Args:
            x: 1D tensor of values to add
        """
        assert x.dim() == 1, f"Expected 1D tensor, got shape {x.shape}"
        x = x.float()  # Ensure float type
        
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        """
        Calculate variance of the running window
        Returns:
            torch.Tensor: variance values, or None if not enough data
        """
        if self.values is None or self.values.shape[1] < 2:  # Need at least 2 points for variance
            return None
            
        try:
            if self.norm:
                # Use unbiased variance and ensure no division by zero
                var = torch.var(self.values, dim=1, unbiased=True)
                denominator = max(self.values.shape[1], 1)
                return var / denominator
            else:
                return torch.var(self.values, dim=1, unbiased=True)
        except RuntimeError as e:
            print(f"Variance calculation error: {e}")
            return None


class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        """
        Args:
            input_ids: tokens generated so far
            scores: scores for each token in vocabulary
        Returns:
            bool: whether to stop generation
        """
        try:
            last_scores = scores[-1]
            # Get maximum probability for each sequence
            max_scores = last_scores.max(1)[0].float().cpu()
            self.vars.push(max_scores)
            
            var = self.vars.variance()
            if var is not None:
                self.varvars.push(var)
            
            self.size += 1
            if self.size < self.window_size:
                return False

            varvar = self.varvars.variance()
            if varvar is None:
                return False
            
            # Check stopping criteria for each batch
            for b in range(len(last_scores)):
                if varvar[b] < self.threshold:
                    if self.stop_inds[b] > 0 and not self.stopped[b]:
                        self.stopped[b] = self.stop_inds[b] >= self.size
                    else:
                        # Calculate new stopping index
                        self.stop_inds[b] = int(
                            min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                        )
                else:
                    # Reset stopping criteria if variance is above threshold
                    self.stop_inds[b] = 0
                    self.stopped[b] = False
            
            # Stop if all sequences in batch meet stopping criteria
            return all(self.stopped.values()) and len(self.stopped) > 0
            
        except Exception as e:
            print(f"Error in stopping criteria: {e}")
            return False
