# deep learning libraries
import torch

# other libraries
from typing import Iterator, Dict, Any, DefaultDict


class SGD(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self, params: Iterator[torch.nn.Parameter], lr=1e-3, weight_decay: float = 0.0
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay)

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO

        for group in self.param_groups:
            for p in group["params"]:

                if p.grad is None:
                    continue
                
                d_p = p.grad

                if group["weigh_decay"] != 0: # Apply L2 regularization
                    d_p = d_p + group["weigh_decay"]*p.data
                
                # Update step
                p.data = p.data - group["lr"]*d_p
                


class SGDMomentum(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        # define defaults
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay, momentum = momentum)

        # call super class constructor
        super().__init__(params, defaults)


    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Attr:
            param_groups: list with the dict of the parameters.
            state: dict with the state for each parameter.
        """

        # TODO
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                # Weight decay
                if group["weight_decay"] != 0:
                    d_p = d_p + group["weight_decay"] * p.data

                # Momentum
                if group["momentum"] != 0:
                    param_state = self.state[p] # acceder al estado del parámetro

                    if "momentum_buffer" not in param_state:
                        # Inicializar momenutum buffer
                        group["momentum_buffer"] = torch.zeros_like(p.data)
                    
                    v = group["momentum_buffer"]

                    # vt =pvt-1 + dp 
                    v = group["momentum"] * v + d_p
                    param_state["momentum_buffer"] = v

                    d_p = v
                
                p.data = p.data - group["lr"] * d_p
                

                    
                        
                    
                




class SGDNesterov(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        defaults: Dict[Any, Any] = dict(lr=lr, momentum=momentum, weight_decay= weight_decay)
        super.__init__(defaults=defaults, params=params)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO

        for group in self.param_groups:
            for p in group["params"]:

                if p.grad is None:
                    continue
                
                d_p = p.grad

                if group["weight_decay"] != 0:
                    d_p += group["weight_decay"]*p.data
                
                # Momentum
                if group["momentum"] != 0:
                    param_state = self.state[p]

                    if "momentum_buffer" not in param_state:
                        # Initialize momentum buffer
                        param_state["momentum_buffer"] = torch.zeros_like(p.data)

                    # Apply nesterov
                    v = group["momentum"] * param_state["momentum_buffer"] + d_p
                    # Guardar v
                    param_state["momentum_buffer"] = v

                    d_p +=  group["momentum"]*v # Corregir una vez estés ahí
                
                p.data -= group["lr"]*d_p 
                


                




class Adam(torch.optim.Optimizer):
    """
    This class is a custom implementation of the Adam algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super.__init__(defaults, params)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
            """
            This method is the step of the optimization algorithm.

            Args:
                closure: Ignore this parameter. Defaults to None.
            """
            for group in self.param_groups:
                for p in group["params"]:
                    
                    if p.grad is None:
                        continue

                    d_p = p.grad

                    if group["weight_decay"] != 0:
                        d_p += group["weight_decay"]*p.data

                    # Adam
                    b1, b2 = group["betas"]

                    param_state = self.state[p]

                    if "momentum_buffer" not in param_state:

                        # In initialize state variables:
                        param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        param_state["momentum_buffer2"] = torch.zeros_like(p.data)
                        param_state["t"] = 1

                    v = param_state["momentum_buffer"]
                    s = param_state["momentum_buffer2"]
                    t = param_state["t"]

                    # Apply Adam:

                    v = b1*v + (1-b1)*d_p
                    s = b2*s + (1-b2)*d_p**2

                    param_state["momentum_buffer"] = v
                    param_state["momentum_buffer2"] = s
                    v_t = v/(1-b1**t)
                    s_t = s/(1-b2**t)

                    param_state["t"] += 1

                    d_p = group["lr"]*v_t/(torch.sqrt(s_t) + group["eps"])

                    p.data -= d_p

class NAdam(torch.optim.Optimizer):
    """
    This class is a custom implementation of the NAdam algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 0.004,
    ) -> None:
        """
        This is the constructor for NAdam.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
            betas: betas for Adam. Defaults to (0.9, 0.999).
            eps: epsilon for approximation. Defaults to 1e-8.
            weight_decay: weight decay. Defaults to 0.0.
            momentum_decay: momentum decay. Defaults to 0.004.
        """

        # TODO
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO
        
        for group in self.param_groups:
            for p in group["params"]:

                if p.grad is None:
                    continue

                d_p = p.grad
                
                if group["weight_decay"] != 0:
                    d_p += group["weight_decay"] * p.data # d_p += lambda*|theta|
                
                # Apply NAdam
                b1, b2 = group["betas"]
                phi = group["momentum_decay"]

                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    param_state["momentum_buffer2"] = torch.zeros_like(p.data)
                    param_state["t"] = 1
                
                v = param_state["momentum_buffer"]
                s = param_state["momentum_buffer2"]
                t = param_state["t"]
    
                mu_t = b1*(1 - 1/2*0.96**(t*phi))
                mu_t_next = b1*(1 - 1/2*0.96**((t+1)*phi))

                v = b1*v + (1-b1) * d_p
                s = b2*s + (1-b2) * d_p**2
                t += 1

                # Guardar momentos
                param_state["momentum_buffer"] = v
                param_state["momentum_buffer2"] = s
                param_state["t"] = t

                mu_prod = 1
                for k in range(t):
                    mu_prod *= b1*(1 - 1/2*0.96**(k*phi))

                mu_prod_next = mu_prod*mu_t_next

                v_t = mu_t_next*v/(1 - mu_prod_next) + (1 - mu_t)*d_p/(1 - mu_prod)
                s_t = s/ (1 - b2**t)

                p.data -= group["lr"] * v_t/(s_t.sqrt() + group["eps"])



        return None



class AdaGrad(torch.optim.Optimizer):
    """
    This class is a custom implementation of the AdaGrad algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-2,
        eps: float = 1e-10,
        weight_decay: float = 0.0,
        decay_rate: float = 0.95,
    ) -> None:
        """
        This is the constructor for AdaGrad.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-2.
            eps: epsilon for numerical stability. Defaults to 1e-10.
            weight_decay: weight decay. Defaults to 0.0.
            decay_rate: decay rate for the moving average. Defaults to 0.95.
        """
        # TODO
        defaults = dict(
            lr = lr,
            eps = eps,
            weight_decay=weight_decay,
            decay_rate=decay_rate,
        )
        
        super().__init__(defaults=defaults, params = params)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """
        # TODO
        for group in self.param_groups:
            for p in group["params"]:
                
                if p.grad is None:
                    continue

                d_p = p.grad
                eta = group["decay_rate"]

                if group["weight_decay"] != 0:
                    # apply l2 reg
                    d_p += group["weight_decay"]*p.data
                
                param_state = self.state[p]
                if "gamma" not in param_state:
                    param_state["gamma"] = torch.zeros_like(p.data)
                    param_state["state_sum"] = torch.zeros_like(p.data)
                    param_state["t"] = 0
                
                t = param_state["t"]
                gamma = param_state["gamma"]/(1 + (t-1)*eta)
                state_sum = param_state["state_sum"] + d_p **2
                t += 1
                
                # Update states
                param_state["gamma"] = gamma
                param_state["t"] = t
                param_state["state_sum"] = state_sum

                p.data -= gamma * d_p/(state_sum.sqrt() + group["eps"])             


        




class AdaDelta(torch.optim.Optimizer):
    """
    This class is a custom implementation of the AdaGrad algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]
    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for AdaDelta.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1.0.
            rho: decay rate for moving average of squared gradients. Defaults to 0.9.
            eps: epsilon for numerical stability. Defaults to 1e-6.
            weight_decay: weight decay. Defaults to 0.0.
        """
        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """
        # TODO
        for group in self.param_groups:
            for p in group["params"]:
                
                if p.grad is None:
                    continue

                d_p = p.grad
                
                if group["weight_decay"] != 0:
                    # apply l2 reg
                    d_p += group["weight_decay"]*p.data
                
                param_state = self.state[p]
                rho = group["rho"]
                eps = group["eps"]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    param_state["momentum_buffer2"] = torch.zeros_like(p.data)
                
                v = param_state["momentum_buffer"]
                v = v*rho + d_p**2*(1 - rho)
                param_state["momentum_buffer"] = v # update vt

                u = param_state["momentum_buffer2"]
                delta_x = (u + eps).sqrt()*d_p/(v + eps).sqrt()
                u = u * rho + delta_x**2 * (1 - rho)
                param_state["momentum_buffer2"] = u

                p.data -= group["lr"] * delta_x
                


        




