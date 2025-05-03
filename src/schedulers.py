"""
This file contains custom implemenation of LR schedulers which are an optimizer wrapper.
"""

# deep learning libraries
import torch

# other libraries
from typing import Optional
import math

class StepLR(torch.optim.lr_scheduler.LRScheduler):
    """
    This

    Attr:
        optimizer: optimizer that the scheduler is using.
        step_size: number of steps to decrease learning rate.
        gamma: factor to decrease learning rate.
        count: count of steps.
    """

    optimizer: torch.optim.Optimizer
    step_size: int
    gamma: float
    last_epoch: int
    counters: int

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float = 0.1,
        
    ) -> None:
        """
        This method is the constructor of StepLR class.

        Args:
            optimizer: optimizer.
            step_size: size of the step.
            gamma: factor to change the lr. Defaults to 0.1.
        """

        # TODO
        self._step_size = step_size
        self._gamma = gamma
        self._counters = 0
        super().__init__(optimizer) # last_epoch of father default to -1

    def step(self, epoch: Optional[int] = None) -> None:
        """
        This function is the step of the scheduler.

        Args:
            epoch: ignore this argument. Defaults to None.
        """

        # TODO
        if (self._counters % self._step_size == 0) and self._counters != 0:
            
            for group in self.optimizer.param_groups: 
                # note: specific lr for each group

                group["lr"] *= self._gamma
        
        self._counters += 1

        return None


class MultiStepLR(torch.optim.lr_scheduler.LRScheduler):
    """
    This class implements the MultiStepLR scheduler.
    Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.

    Attr:
        optimizer: optimizer that the scheduler is using.
        milestones: list of step indices where the lr will be reduced.
        gamma: factor to reduce the learning rate.
        counters: step counter to track progress.
    """

    optimizer: torch.optim.Optimizer
    milestones: list[int]
    gamma: float
    counters: int

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: list[int],
        gamma: float = 0.1,
    ) -> None:
        """
        This method is the constructor of MultiStepLR class.

        Args:
            optimizer: optimizer to wrap.
            milestones: list of steps at which to reduce lr.
            gamma: multiplicative factor for lr reduction.
        """

        # TODO
        self._milestones = milestones
        self._gamma = gamma
        self._counters = 0
        super().__init__(optimizer=optimizer)
        


    def step(self, epoch: Optional[int] = None) -> None:
        """
        This function is the step of the scheduler.

        Args:
            epoch: ignore this argument. Defaults to None.
        """

        # TODO
        if self._counters in self._milestones:
            for group in self.optimizer.param_groups:
                group["lr"] *= self._gamma
        
        self._counters += 1



class CosineAnnealingLR(torch.optim.lr_scheduler.LRScheduler):
    """
    This class implements the CosineAnnealingLR scheduler.

    Attr:
        optimizer: optimizer that the scheduler is using.
        T_max: maximum number of iterations.
        eta_min: minimum learning rate.
        counters: step counter to track current iteration.
    """

    optimizer: torch.optim.Optimizer
    T_max: int
    eta_min: float
    counters: int

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0.0,
    ) -> None:
        """
        This method is the constructor of CosineAnnealingLR class.

        Args:
            optimizer: optimizer to wrap.
            T_max: maximum number of iterations.
            eta_min: minimum learning rate. Defaults to 0.0.
        """

        # TODO
        self.T_max = T_max
        self._eta_min = eta_min
        self.counters = 0
        super().__init__(optimizer=optimizer, last_epoch=-1)

    def step(self, epoch: Optional[int] = None) -> None:
        """
        This function updates the learning rate using the cosine annealing schedule.

        Args:
            epoch: ignored. Defaults to None.
        """

        # TODO
                    
 
        if self.counters <= self.T_max:
            for i,group in enumerate(self.optimizer.param_groups):
                    
                eta_max = self.base_lrs[i]
                next_eta = self._eta_min + 1/2*(eta_max - self._eta_min)*(
                    1 + math.cos(self.counters*math.pi/self.T_max))

                group["lr"] = next_eta

            else:
                for group in self.optimizer.param_groups:
                    group["lr"] = self._eta_min

        self.counters += 1   


class CyclicLR(torch.optim.lr_scheduler.LRScheduler):
    """
    This class implements the CyclicLR scheduler.

    Attr:
        optimizer: optimizer that the scheduler is using.
        base_lr: minimum learning rate.
        max_lr: maximum learning rate.
        step_size_up: number of steps to reach max_lr.
        mode: cyclic policy (e.g., "triangular").
        counters: step counter.
    """

    optimizer: torch.optim.Optimizer
    base_lr: float
    max_lr: float
    step_size_up: int
    mode: str
    counters: int

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        mode: str = "triangular",
    ) -> None:
        """
        This method is the constructor of CyclicLR class.

        Args:
            optimizer: optimizer to wrap.
            base_lr: initial learning rate.
            max_lr: upper learning rate bound.
            step_size_up: steps to increase lr. Defaults to 2000.
            mode: policy mode. Defaults to 'triangular'.
        """

        # TODO

    def step(self, epoch: Optional[int] = None) -> None:
        """
        This function updates the learning rate using the cyclic schedule.

        Args:
            epoch: ignored. Defaults to None.
        """

        # TODO
