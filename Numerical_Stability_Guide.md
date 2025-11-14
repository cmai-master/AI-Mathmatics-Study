# Numerical Stability in Deep Learning

*A practical guide to numerical issues and solutions in AI/ML implementation*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Floating Point Arithmetic](#2-floating-point-arithmetic)
3. [Common Numerical Issues](#3-common-numerical-issues)
4. [Stable Implementations](#4-stable-implementations)
5. [Mixed Precision Training](#5-mixed-precision-training)
6. [Debugging Numerical Issues](#6-debugging-numerical-issues)

---

## 1. Introduction

### Why Numerical Stability Matters

In deep learning, we often encounter:
- **Very large numbers**: e^(100) ≈ 2.7 × 10^43
- **Very small numbers**: e^(-100) ≈ 3.7 × 10^(-44)
- **Operations on these**: Can cause overflow, underflow, or loss of precision

**Real-world consequences:**
- Training diverges (NaN losses)
- Gradients vanish or explode
- Model produces incorrect predictions
- Code crashes

### Learning Objectives

✅ Understand floating point representation
✅ Recognize common numerical problems
✅ Implement stable versions of key operations
✅ Use mixed precision training safely
✅ Debug numerical issues in neural networks

---

## 2. Floating Point Arithmetic

### 2.1 IEEE 754 Standard

**Single Precision (FP32)**:
```
Sign (1 bit) | Exponent (8 bits) | Mantissa (23 bits)
Total: 32 bits

Range: ±1.4 × 10^(-45) to ±3.4 × 10^(38)
Precision: ~7 decimal digits
```

**Half Precision (FP16)**:
```
Sign (1 bit) | Exponent (5 bits) | Mantissa (10 bits)
Total: 16 bits

Range: ±6.0 × 10^(-8) to ±6.5 × 10^(4)
Precision: ~3 decimal digits
```

**Double Precision (FP64)**:
```
Sign (1 bit) | Exponent (11 bits) | Mantissa (52 bits)
Total: 64 bits

Range: ±4.9 × 10^(-324) to ±1.8 × 10^(308)
Precision: ~16 decimal digits
```

### 2.2 Key Properties

**Machine Epsilon** (ε):
- FP32: ε ≈ 1.19 × 10^(-7)
- FP16: ε ≈ 9.77 × 10^(-4)
- FP64: ε ≈ 2.22 × 10^(-16)

Meaning: Smallest number such that `1.0 + ε ≠ 1.0` in floating point

**Implications**:
```python
# FP32 example
x = 1.0
y = 1e-8
result = x + y  # result == 1.0 (y is too small!)
```

### 2.3 Catastrophic Cancellation

Problem: Subtracting nearly equal numbers loses precision

**Example**:
```python
# Bad: catastrophic cancellation
a = 1.23456789e8
b = 1.23456788e8
c = a - b  # Should be 1000, but gets ~1024 due to rounding

# Better: reformulate mathematically
# Instead of (a - b), use algebraic identity if possible
```

---

## 3. Common Numerical Issues

### 3.1 Overflow

**Definition**: Number exceeds maximum representable value → `inf`

**Common causes in ML**:
```python
# 1. Exponential without normalization
exp(100)  # = inf in FP32

# 2. Large dot products
x = np.array([100.0] * 1000)
np.dot(x, x)  # = inf

# 3. Gradient explosion
# Gradients multiply through layers
```

**Solutions**:
- Normalize inputs
- Use log-space computations
- Gradient clipping
- Careful initialization

### 3.2 Underflow

**Definition**: Number too small to represent → 0

**Common causes**:
```python
# 1. Product of many small numbers
prob = 0.001 ** 1000  # = 0.0 (underflow)

# 2. Vanishing gradients
# sigmoid derivative: σ'(x) = σ(x)(1-σ(x))
# For large |x|, σ'(x) ≈ 0
```

**Solutions**:
- Log probabilities instead of probabilities
- Activation functions (ReLU instead of sigmoid)
- Batch normalization
- Residual connections

### 3.3 Loss of Precision

**Associativity fails**:
```python
# Floating point addition is NOT associative
(1e20 + 1.0) - 1e20  # = 0.0
1e20 + (1.0 - 1e20)  # = 1.0

# Summation order matters!
```

**Better summation** (Kahan algorithm):
```python
def kahan_sum(numbers):
    """
    Numerically stable summation
    """
    s = 0.0
    c = 0.0  # Compensation for lost low-order bits

    for x in numbers:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t

    return s
```

---

## 4. Stable Implementations

### 4.1 Softmax

**Naive implementation** (UNSTABLE):
```python
def softmax_naive(x):
    """
    Numerically UNSTABLE!
    """
    return np.exp(x) / np.sum(np.exp(x))

# Problem:
x = np.array([1000, 1001, 1002])
softmax_naive(x)  # = [nan, nan, nan] (overflow!)
```

**Stable implementation**:
```python
def softmax_stable(x):
    """
    Numerically STABLE softmax

    Trick: softmax(x) = softmax(x - c) for any c
    Choose c = max(x) to prevent overflow
    """
    # Shift by max for numerical stability
    x_shifted = x - np.max(x)

    # Now all exponents are ≤ 0, no overflow
    exp_x = np.exp(x_shifted)

    return exp_x / np.sum(exp_x)

# Mathematical justification:
# softmax(x) = exp(x_i) / Σ exp(x_j)
#            = exp(x_i - c) / Σ exp(x_j - c)  [multiply by exp(-c)/exp(-c)]
#            = softmax(x - c)
```

**Vectorized version**:
```python
def softmax_stable_batch(X, axis=-1):
    """
    Stable softmax for batches

    Args:
        X: array of shape (batch_size, num_classes)
        axis: axis along which to compute softmax
    """
    X_shifted = X - np.max(X, axis=axis, keepdims=True)
    exp_X = np.exp(X_shifted)
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)
```

### 4.2 Log-Sum-Exp

**Problem**: Computing log(Σ exp(xᵢ)) appears everywhere:
- Softmax normalization
- Log-likelihood computations
- Logsumexp in probabilistic models

**Naive implementation** (UNSTABLE):
```python
def logsumexp_naive(x):
    """
    Numerically UNSTABLE!
    """
    return np.log(np.sum(np.exp(x)))

# Problem:
x = np.array([1000, 1001, 1002])
logsumexp_naive(x)  # = inf
```

**Stable implementation**:
```python
def logsumexp_stable(x):
    """
    Stable log-sum-exp

    log(Σ exp(x_i)) = log(Σ exp(x_i - c) * exp(c))
                    = log(exp(c) * Σ exp(x_i - c))
                    = c + log(Σ exp(x_i - c))

    Choose c = max(x)
    """
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))

# Verify:
x = np.array([1000, 1001, 1002])
print(logsumexp_stable(x))  # ~1002.407606 ✓

# scipy.special.logsumexp uses this trick
from scipy.special import logsumexp
print(logsumexp(x))  # Same result
```

### 4.3 Log Probability Computations

**Never use raw probabilities**:
```python
# BAD: underflow
probs = [0.001] * 1000
total_prob = np.prod(probs)  # = 0.0 (underflow)

# GOOD: use log probabilities
log_probs = [np.log(0.001)] * 1000
total_log_prob = np.sum(log_probs)  # = -6907.76 ✓
total_prob = np.exp(total_log_prob)  # Convert back if needed
```

**Log-domain operations**:
```python
# Addition in log space: log(a + b) = ?
def log_add(log_a, log_b):
    """
    Compute log(a + b) given log(a) and log(b)
    """
    if log_a > log_b:
        return log_a + np.log1p(np.exp(log_b - log_a))
    else:
        return log_b + np.log1p(np.exp(log_a - log_b))

# np.log1p(x) = log(1 + x), but more accurate for small x
```

### 4.4 Batch Normalization Variance

**Naive implementation** (UNSTABLE):
```python
def batch_norm_naive(x, eps=1e-5):
    """
    Numerically UNSTABLE for large values!
    """
    mean = np.mean(x, axis=0)
    var = np.mean((x - mean)**2, axis=0)  # Two passes

    # Problem: var can be negative due to rounding!
    # Also: Var(X) = E[X²] - E[X]² is catastrophic cancellation

    return (x - mean) / np.sqrt(var + eps)
```

**Stable implementation**:
```python
def batch_norm_stable(x, eps=1e-5):
    """
    Numerically STABLE batch normalization

    Use Welford's online algorithm or two-pass method
    """
    mean = np.mean(x, axis=0)

    # Two-pass is more stable than one-pass variance formula
    var = np.var(x, axis=0)  # NumPy uses two-pass internally

    # Clamp variance to be non-negative
    var = np.maximum(var, 0.0)

    return (x - mean) / np.sqrt(var + eps)

# Even better: Welford's online algorithm
def welford_variance(x):
    """
    Welford's online algorithm for variance
    More numerically stable than any batch method
    """
    n = 0
    mean = 0.0
    M2 = 0.0

    for x_i in x:
        n += 1
        delta = x_i - mean
        mean += delta / n
        delta2 = x_i - mean
        M2 += delta * delta2

    variance = M2 / n if n > 0 else 0.0
    return mean, variance
```

### 4.5 Gradient Clipping

**Why clip**:
- Prevent gradient explosion
- Stabilize training
- Enable larger learning rates

**Norm clipping**:
```python
def clip_grad_norm(gradients, max_norm):
    """
    Clip gradients by global norm

    If ||g|| > max_norm, scale to g := g * max_norm / ||g||
    """
    # Compute global norm
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))

    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        gradients = [g * clip_coef for g in gradients]

    return gradients

# PyTorch equivalent:
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

**Value clipping**:
```python
def clip_grad_value(gradients, clip_value):
    """
    Clip each gradient element to [-clip_value, clip_value]
    """
    return [np.clip(g, -clip_value, clip_value) for g in gradients]

# PyTorch equivalent:
# torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
```

---

## 5. Mixed Precision Training

### 5.1 Motivation

**Why use FP16**:
- ✅ 2x faster on modern GPUs (Tensor Cores)
- ✅ 2x less memory (fit larger models/batches)
- ✅ 2x faster data transfer

**Challenges**:
- ❌ Limited range: overflow/underflow
- ❌ Low precision: accumulation errors
- ❌ Gradient underflow (most critical!)

### 5.2 Loss Scaling

**Problem**: Gradients often in range [1e-8, 1e-6] → underflow in FP16

**Solution**: Scale loss before backward pass

```python
# PyTorch example
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward in FP16
    with torch.cuda.amp.autocast():
        output = model(batch)
        loss = criterion(output, target)

    # Scale loss before backward
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()  # Gradients are scaled

    # Unscale before optimizer step
    scaler.step(optimizer)  # Unscales gradients internally
    scaler.update()  # Update scale for next iteration
```

**Dynamic loss scaling**:
```python
class DynamicLossScaler:
    def __init__(self, init_scale=2**16, scale_factor=2, scale_window=2000):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.iter = 0
        self.last_overflow = 0

    def update(self, overflow):
        self.iter += 1

        if overflow:
            # Overflow detected, reduce scale
            self.scale /= self.scale_factor
            self.last_overflow = self.iter
        elif (self.iter - self.last_overflow) >= self.scale_window:
            # No overflow for a while, increase scale
            self.scale *= self.scale_factor
```

### 5.3 Master Weights in FP32

**Best practice**: Keep optimizer states in FP32

```python
# Model weights
weights_fp16 = model.parameters()  # FP16 for forward/backward

# Optimizer maintains FP32 copy
optimizer_fp32_weights = [w.float() for w in weights_fp16]

# Training step:
# 1. Forward/backward in FP16
# 2. Update FP32 weights
# 3. Copy back to FP16

# PyTorch does this automatically with AMP
```

### 5.4 Safe Mixed Precision

**Operations that should stay in FP32**:
- Batch normalization (variance computation)
- Softmax
- Loss computation
- Small activations that might underflow

**PyTorch automatic mixed precision (AMP)**:
```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        # Automatic mixed precision context
        with autocast():
            # Forward pass (FP16 where safe)
            output = model(input)
            loss = criterion(output, target)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Optimizer step (unscales gradients)
        scaler.step(optimizer)
        scaler.update()
```

---

## 6. Debugging Numerical Issues

### 6.1 Detecting NaN/Inf

**Check for NaN**:
```python
import numpy as np
import torch

# NumPy
if np.isnan(x).any():
    print("NaN detected!")

if np.isinf(x).any():
    print("Inf detected!")

# PyTorch
if torch.isnan(x).any():
    print("NaN detected!")

# Set to detect NaN automatically
torch.autograd.set_detect_anomaly(True)
```

**Find where NaN first appears**:
```python
# Register hooks to detect NaN in gradients
def check_nan_hook(grad):
    if torch.isnan(grad).any():
        raise ValueError(f"NaN gradient detected!")

for param in model.parameters():
    param.register_hook(check_nan_hook)
```

### 6.2 Gradient Checking

**Numerical gradient**:
```python
def numerical_gradient(f, x, eps=1e-5):
    """
    Compute numerical gradient using finite differences

    grad_i ≈ (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
    """
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps

        x_minus = x.copy()
        x_minus[i] -= eps

        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)

    return grad

# Check backprop implementation
analytical_grad = compute_backprop_gradient(...)
numerical_grad = numerical_gradient(loss_fn, params)

relative_error = np.linalg.norm(analytical_grad - numerical_grad) / \
                 (np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad))

if relative_error < 1e-7:
    print("✓ Gradient implementation correct!")
elif relative_error < 1e-4:
    print("⚠ Gradient might have issues")
else:
    print("✗ Gradient implementation wrong!")
```

### 6.3 Common Fixes

**Issue**: Training loss becomes NaN

**Checklist**:
1. ✅ Check learning rate (too large?)
2. ✅ Check gradient norms (exploding?)
3. ✅ Check input data (contains NaN/Inf?)
4. ✅ Check weight initialization
5. ✅ Add gradient clipping
6. ✅ Use batch normalization
7. ✅ Switch to more stable optimizer (Adam instead of SGD)

**Issue**: Gradients vanish to zero

**Solutions**:
1. ✅ Use ReLU instead of sigmoid/tanh
2. ✅ Add residual connections
3. ✅ Use batch normalization
4. ✅ Check weight initialization (Xavier, He init)
5. ✅ Reduce network depth
6. ✅ Increase learning rate slightly

**Issue**: Loss stuck (not decreasing)

**Possible causes**:
1. Learning rate too small
2. Weights initialized poorly
3. Dead ReLUs
4. Batch size too small/large
5. Data not shuffled

---

## 7. Best Practices Summary

### ✅ DO:

1. **Always use stable implementations**
   - Softmax: subtract max before exp
   - Logsumexp: shift by max
   - Use log probabilities for likelihoods

2. **Normalize inputs**
   - Zero mean, unit variance
   - Or scale to [0, 1] or [-1, 1]

3. **Initialize weights carefully**
   - Xavier/Glorot for tanh/sigmoid
   - He initialization for ReLU
   - Orthogonal for RNNs

4. **Clip gradients**
   - Norm clipping for transformers
   - Value clipping for RL

5. **Use appropriate precision**
   - FP32 for small models
   - Mixed precision for large models
   - FP64 for debugging only

6. **Monitor training**
   - Log gradient norms
   - Log activation statistics
   - Check for NaN/Inf regularly

### ❌ DON'T:

1. **Never compute**:
   - Variance as E[X²] - E[X]²
   - Probabilities as product of many small numbers
   - Exponentials without normalization

2. **Avoid**:
   - Very large learning rates
   - Extreme weight initializations
   - Unnormalized inputs
   - Deep networks without residual connections

3. **Don't ignore**:
   - Warning messages about numerical issues
   - Gradient explosion/vanishing
   - Sudden loss spikes

---

## 8. Quick Reference

### Stable Softmax
```python
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
```

### Stable Log-Sum-Exp
```python
def logsumexp(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))
```

### Stable Cross-Entropy
```python
def cross_entropy(logits, labels):
    # Stable: log(softmax(x)) = x - logsumexp(x)
    log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
    return -np.mean(np.sum(labels * log_probs, axis=-1))
```

### Gradient Clipping
```python
# PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# NumPy
norm = np.linalg.norm(grad)
if norm > max_norm:
    grad = grad * max_norm / norm
```

### Mixed Precision Training
```python
# PyTorch AMP
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## References

1. **Goldberg, D.** (1991). *What every computer scientist should know about floating-point arithmetic*. ACM Computing Surveys, 23(1), 5-48.

2. **Micikevicius, P., et al.** (2018). *Mixed Precision Training*. ICLR 2018.

3. **Higham, N. J.** (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.

4. **PyTorch Documentation**: Automatic Mixed Precision
   - https://pytorch.org/docs/stable/amp.html

5. **TensorFlow Mixed Precision Guide**
   - https://www.tensorflow.org/guide/mixed_precision

---

**Next**: [Advanced Topics 2024](./Advanced_Topics_2024.md) - Modern AI Mathematics
