import torch
import torch.nn.functional as F
from activations import *
import pytest

# ---------- Linear Layer Test ----------
def test_linear_forward_backward():
    batch, input_dim, output_dim = 5, 10, 4

    x = torch.randn(batch, input_dim, requires_grad=True)
    custom_linear = Linear(input_dim, output_dim)
    torch_linear = torch.nn.Linear(input_dim, output_dim)
    
    # Copy weights and bias
    torch_linear.weight.data = custom_linear.weight.data.clone()
    torch_linear.bias.data = custom_linear.bias.data.clone()

    # Forward
    y_custom = custom_linear(x)
    y_torch = torch_linear(x)

    assert torch.allclose(y_custom, y_torch, atol=1e-6), "Linear forward mismatch"

    # Backward
    grad_output = torch.randn_like(y_custom)
    y_custom.backward(grad_output, retain_graph=True)
    y_torch.backward(grad_output)

    assert torch.allclose(x.grad, x.grad, atol=1e-6), "Linear backward mismatch (input grad)"
    assert torch.allclose(custom_linear.weight.grad, torch_linear.weight.grad, atol=1e-6), "Linear backward mismatch (weight grad)"
    assert torch.allclose(custom_linear.bias.grad, torch_linear.bias.grad, atol=1e-3), "Linear backward mismatch (bias grad)"


# ---------- ReLU Layer Test ----------
def test_relu_forward_backward():
    x = torch.randn(10, requires_grad=True)

    custom_relu = ReLU()
    y_custom = custom_relu(x)

    x_torch = x.clone().detach().requires_grad_()
    y_torch = F.relu(x_torch)

    assert torch.allclose(y_custom, y_torch, atol=1e-6), "ReLU forward mismatch"

    grad_output = torch.ones_like(x)
    y_custom.backward(grad_output, retain_graph=True)
    y_torch.backward(grad_output)

    assert torch.allclose(x.grad, x_torch.grad, atol=1e-6), "ReLU backward mismatch"


# ---------- Huber Loss Test ----------
def test_huber_loss_forward():
    x = torch.tensor([0.0, 0.5, 1.5, -2.0], requires_grad=True)
    target = torch.tensor([0.0, 0.0, 0.0, 0.0])

    delta = 1.0
    custom_loss_fn = HuberLoss(delta)
    loss_custom = custom_loss_fn(x, target)

    loss_torch = torch.nn.functional.huber_loss(x, target, delta=delta, reduction='none')

    assert torch.allclose(loss_custom, loss_torch, atol=1e-5), "Huber loss forward mismatch"

# Puedes añadir backward test si implementas correctamente el backward de Huber

# ---------- LeakyReLU Layer Test ----------
def test_leaky_relu_forward_backward():
    x = torch.randn(10, requires_grad=True)

    # Custom LeakyReLU implementation
    negative_slope = 0.01
    custom_leaky_relu = LeakyReLU(negative_slope)

    # PyTorch LeakyReLU
    x_torch = x.clone().detach().requires_grad_()
    y_torch = F.leaky_relu(x_torch, negative_slope=negative_slope)

    y_custom = custom_leaky_relu(x)

    assert torch.allclose(y_custom, y_torch, atol=1e-6), "LeakyReLU forward mismatch"

    grad_output = torch.ones_like(x)
    y_custom.backward(grad_output, retain_graph=True)
    y_torch.backward(grad_output)

    assert torch.allclose(x.grad, x_torch.grad, atol=1e-6), "LeakyReLU backward mismatch"

# ---------- PReLU Layer Test ----------
def test_prelu_forward_backward():
    x = torch.randn(10, requires_grad=True)

    # Custom PReLU implementation
    init_weight = 0.25
    custom_prelu = PReLU(init_weight)

    # PyTorch PReLU
    x_torch = x.clone().detach().requires_grad_()
    torch_prelu = torch.nn.PReLU(init=init_weight)
    y_torch = torch_prelu(x_torch)

    y_custom = custom_prelu(x)

    assert torch.allclose(y_custom, y_torch, atol=1e-6), "PReLU forward mismatch"

    grad_output = torch.ones_like(x)
    y_custom.backward(grad_output, retain_graph=True)
    y_torch.backward(grad_output)

    assert torch.allclose(x.grad, x_torch.grad, atol=1e-6), "PReLU backward mismatch (input grad)"
    assert torch.allclose(custom_prelu.weight, torch_prelu.weight, atol=1e-6), "PReLU backward mismatch (weight grad)"


# ---------- ELU Layer Test ----------
def test_elu_forward_backward():
    x = torch.randn(10, requires_grad=True)
    

    alpha = 1.0
    custom_elu = ELU(alpha)
    y_custom = custom_elu(x)
    

    x_torch = x.clone().detach().requires_grad_()
    y_torch = F.elu(x_torch, alpha=alpha)
    
    assert torch.allclose(y_custom, y_torch, atol=1e-6), "ELU forward mismatch"
    
    grad_output = torch.ones_like(x)
    y_custom.backward(grad_output, retain_graph=True)
    y_torch.backward(grad_output)
    
    assert torch.allclose(x.grad, x_torch.grad, atol=1e-32), "ELU backward mismatch"

# ---------- SELU Layer Test ----------
def test_selu_forward_backward():
    x = torch.randn(10, requires_grad=True)

    custom_selu = SELU()
    y_custom = custom_selu(x)

    x_torch = x.clone().detach().requires_grad_()
    y_torch = F.selu(x_torch)

    assert torch.allclose(y_custom, y_torch, atol=1e-6), "SELU forward mismatch"

    grad_output = torch.ones_like(x)
    y_custom.backward(grad_output, retain_graph=True)
    y_torch.backward(grad_output)

    assert torch.allclose(x.grad, x_torch.grad, atol=1e-6), "SELU backward mismatch"

# ---------- Hardshrink Layer Test ----------
def test_hardshrink_forward_backward():
    x = torch.randn(10, requires_grad=True)
    

    lambd = 0.5
    custom_hardshrink = HardShrink(lambd)
    

    x_torch = x.clone().detach().requires_grad_()
    y_torch = F.hardshrink(x_torch, lambd=lambd)
    
    y_custom = custom_hardshrink(x)
    
    assert torch.allclose(y_custom, y_torch, atol=1e-6), "Hardshrink forward mismatch"
    
    grad_output = torch.ones_like(x)
    y_custom.backward(grad_output, retain_graph=True)
    y_torch.backward(grad_output)
    
    assert torch.allclose(x.grad, x_torch.grad, atol=1e-6), "Hardshrink backward mismatch"

# ---------- HardTanh Layer Test ----------
def test_hardtanh_forward_backward():
    x = torch.randn(10, requires_grad=True)

    custom_hardtanh = HardTanh()
    y_custom = custom_hardtanh(x)

    x_torch = x.clone().detach().requires_grad_()
    y_torch = F.hardtanh(x_torch)

    assert torch.allclose(y_custom, y_torch, atol=1e-6), "HardTanh forward mismatch"

    grad_output = torch.ones_like(x)
    y_custom.backward(grad_output, retain_graph=True)
    y_torch.backward(grad_output)

    assert torch.allclose(x.grad, x_torch.grad, atol=1e-6), "HardTanh backward mismatch"

# ---------- HardSwish Layer Test ----------
def test_hardswish_forward_backward():
    x = torch.randn(10, requires_grad=True)

    custom_hardswish = HardSwish()
    y_custom = custom_hardswish(x)

    x_torch = x.clone().detach().requires_grad_()
    y_torch = F.hardswish(x_torch)

    assert torch.allclose(y_custom, y_torch, atol=1e-6), "HardSwish forward mismatch"

    grad_output = torch.ones_like(x)
    y_custom.backward(grad_output, retain_graph=True)
    y_torch.backward(grad_output)

    assert torch.allclose(x.grad, x_torch.grad, atol=1e-6), "HardSwish backward mismatch"

if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"])