# AI Mathematics Study Guide (Graduate Level)
*A comprehensive guide to mathematical foundations of Artificial Intelligence*

---

## Table of Contents

1. [Linear Algebra for AI](#1-linear-algebra-for-ai)
2. [Probability Theory and Statistics](#2-probability-theory-and-statistics)
3. [Optimization Theory](#3-optimization-theory)
4. [Information Theory](#4-information-theory)
5. [Statistical Learning Theory](#5-statistical-learning-theory)
6. [Deep Learning Mathematics](#6-deep-learning-mathematics)
7. [Advanced Topics](#7-advanced-topics)

---

## 1. Linear Algebra for AI

### 1.1 Vector Spaces and Subspaces

**Definition (Vector Space):** A vector space V over field F is a set with two operations:
- Vector addition: V × V → V
- Scalar multiplication: F × V → V

satisfying 8 axioms (associativity, commutativity, identity, inverse, distributivity).

**Key Concepts:**
- **Subspace:** A subset W ⊆ V that is closed under vector addition and scalar multiplication
- **Span:** span(S) = {∑ αᵢvᵢ : αᵢ ∈ F, vᵢ ∈ S}
- **Linear Independence:** Vectors v₁, ..., vₙ are linearly independent if ∑ αᵢvᵢ = 0 ⟹ αᵢ = 0 ∀i
- **Basis:** A linearly independent spanning set
- **Dimension:** dim(V) = |B| for any basis B of V

### 1.2 Matrix Theory

**Matrix Decompositions:**

1. **Eigendecomposition:**
   - If A ∈ ℝⁿˣⁿ is diagonalizable: A = QΛQ⁻¹
   - Eigenvalues λ satisfy: det(A - λI) = 0
   - Spectral theorem: Symmetric matrices have orthonormal eigenvectors

2. **Singular Value Decomposition (SVD):**
   ```
   A = UΣVᵀ
   where:
   - U ∈ ℝᵐˣᵐ: left singular vectors (eigenvectors of AAᵀ)
   - Σ ∈ ℝᵐˣⁿ: singular values (σᵢ = √λᵢ)
   - V ∈ ℝⁿˣⁿ: right singular vectors (eigenvectors of AᵀA)
   ```

   **Applications:**
   - Principal Component Analysis (PCA)
   - Matrix approximation: Aₖ = ∑ᵢ₌₁ᵏ σᵢuᵢvᵢᵀ
   - Pseudoinverse: A⁺ = VΣ⁺Uᵀ

3. **QR Decomposition:**
   - A = QR, where Q is orthogonal, R is upper triangular
   - Used in: Gram-Schmidt process, least squares, eigenvalue algorithms

4. **Cholesky Decomposition:**
   - For positive definite A: A = LLᵀ
   - Used in: Gaussian processes, optimization

### 1.3 Norms and Inner Products

**Vector Norms:**
- ℓₚ norm: ‖x‖ₚ = (∑ᵢ |xᵢ|ᵖ)^(1/p)
- ℓ₁ (Manhattan): ‖x‖₁ = ∑ᵢ |xᵢ| (sparsity-inducing)
- ℓ₂ (Euclidean): ‖x‖₂ = √(∑ᵢ xᵢ²)
- ℓ∞ (Max): ‖x‖∞ = maxᵢ |xᵢ|

**Matrix Norms:**
- Frobenius: ‖A‖_F = √(∑ᵢⱼ aᵢⱼ²) = √tr(AᵀA)
- Spectral (Operator): ‖A‖₂ = σₘₐₓ(A)
- Nuclear: ‖A‖_* = ∑ᵢ σᵢ (trace norm, convex envelope of rank)

**Inner Products:**
- Standard: ⟨x, y⟩ = xᵀy
- General: ⟨x, y⟩_A = xᵀAy for positive definite A
- Reproducing Kernel Hilbert Space (RKHS): k(x, y) = ⟨φ(x), φ(y)⟩

### 1.4 Matrix Calculus

**Derivatives:**
- ∂(xᵀa)/∂x = a
- ∂(xᵀAx)/∂x = (A + Aᵀ)x = 2Ax (if A symmetric)
- ∂(aᵀXb)/∂X = abᵀ
- ∂tr(AX)/∂X = Aᵀ
- ∂log|X|/∂X = X⁻ᵀ (for invertible X)

**Jacobian and Hessian:**
- Jacobian: J = [∂fᵢ/∂xⱼ]
- Hessian: H = [∂²f/∂xᵢ∂xⱼ]
- For convex f: H ⪰ 0 (positive semidefinite)

### 1.5 Advanced Topics

**Tensor Operations:**
- Tensor product: A ⊗ B
- Kronecker product properties
- Tucker decomposition
- Tensor networks in deep learning

**Random Matrix Theory:**
- Wishart distribution: W ~ W(Σ, n)
- Marchenko-Pastur law
- Applications in covariance estimation

---

## 2. Probability Theory and Statistics

### 2.1 Measure-Theoretic Probability

**Probability Space (Ω, ℱ, P):**
- Sample space: Ω
- σ-algebra: ℱ (closed under countable unions, complements)
- Probability measure: P: ℱ → [0,1]

**Random Variables:**
- Measurable function X: Ω → ℝ
- CDF: F(x) = P(X ≤ x)
- PDF (if exists): f(x) = dF/dx

### 2.2 Important Distributions

**Continuous Distributions:**

1. **Gaussian/Normal:**
   ```
   N(μ, σ²): f(x) = (1/√(2πσ²))exp(-(x-μ)²/(2σ²))

   Multivariate: N(μ, Σ)
   f(x) = (1/√((2π)ⁿ|Σ|))exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
   ```

2. **Exponential Family:**
   ```
   f(x|θ) = h(x)exp(ηᵀT(x) - A(η))
   where:
   - η: natural parameter
   - T(x): sufficient statistic
   - A(η): log-partition function
   ```

   **Members:** Gaussian, Bernoulli, Exponential, Gamma, Beta, Dirichlet

3. **Laplace:**
   - f(x|μ,b) = (1/2b)exp(-|x-μ|/b)
   - Used in: ℓ₁ regularization (Lasso)

**Discrete Distributions:**
- Bernoulli, Binomial, Multinomial
- Poisson: P(X=k) = (λᵏe⁻λ)/k!
- Categorical

### 2.3 Limit Theorems

**Law of Large Numbers:**
- Weak: X̄ₙ →ᵖ μ
- Strong: X̄ₙ →ᵃ·ˢ· μ

**Central Limit Theorem:**
```
√n(X̄ₙ - μ)/σ →ᵈ N(0,1)
```

**Applications:**
- Confidence intervals
- Hypothesis testing
- Bootstrap methods

### 2.4 Bayesian Statistics

**Bayes' Theorem:**
```
p(θ|D) = p(D|θ)p(θ)/p(D)

Posterior ∝ Likelihood × Prior
```

**Conjugate Priors:**
- Beta-Binomial
- Dirichlet-Multinomial
- Gaussian-Gaussian
- Gamma-Poisson

**Bayesian Inference:**

1. **Maximum A Posteriori (MAP):**
   ```
   θ̂_MAP = argmax_θ p(θ|D) = argmax_θ [log p(D|θ) + log p(θ)]
   ```

2. **Posterior Predictive:**
   ```
   p(x*|D) = ∫ p(x*|θ)p(θ|D)dθ
   ```

**Sampling Methods:**
- Markov Chain Monte Carlo (MCMC)
- Metropolis-Hastings
- Gibbs Sampling
- Hamiltonian Monte Carlo
- Variational Inference

### 2.5 Concentration Inequalities

**Markov's Inequality:**
```
P(X ≥ a) ≤ E[X]/a  for a > 0, X ≥ 0
```

**Chebyshev's Inequality:**
```
P(|X - μ| ≥ kσ) ≤ 1/k²
```

**Hoeffding's Inequality:**
```
P(|X̄ - μ| ≥ ε) ≤ 2exp(-2nε²/(b-a)²)
for X_i ∈ [a,b]
```

**Bernstein's Inequality:**
```
P(|X̄ - μ| ≥ ε) ≤ 2exp(-nε²/(2σ² + 2bε/3))
```

**McDiarmid's Inequality:**
- Bounded differences condition
- Applications in learning theory

---

## 3. Optimization Theory

### 3.1 Convex Analysis

**Convex Sets:**
- S is convex if: ∀x,y ∈ S, λ ∈ [0,1]: λx + (1-λ)y ∈ S
- Examples: Hyperplanes, halfspaces, balls, polyhedra

**Convex Functions:**
- f is convex if: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
- Equivalent (if differentiable): f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)
- Equivalent (if twice differentiable): ∇²f(x) ⪰ 0

**Operations Preserving Convexity:**
- Non-negative weighted sum
- Composition with affine mapping
- Pointwise maximum
- Partial minimization

### 3.2 Optimality Conditions

**Unconstrained Optimization:**

First-order necessary: ∇f(x*) = 0

Second-order necessary: ∇²f(x*) ⪰ 0

Second-order sufficient: ∇²f(x*) ≻ 0 ⟹ x* is strict local minimum

**Constrained Optimization:**

**KKT Conditions** for:
```
minimize f(x)
subject to gᵢ(x) ≤ 0, i=1,...,m
           hⱼ(x) = 0, j=1,...,p
```

1. Stationarity: ∇f(x*) + ∑λᵢ∇gᵢ(x*) + ∑νⱼ∇hⱼ(x*) = 0
2. Primal feasibility: gᵢ(x*) ≤ 0, hⱼ(x*) = 0
3. Dual feasibility: λᵢ ≥ 0
4. Complementary slackness: λᵢgᵢ(x*) = 0

### 3.3 Gradient Descent Methods

**Gradient Descent:**
```
x_{k+1} = x_k - α_k∇f(x_k)
```

**Convergence Rate:**
- Lipschitz continuous gradient: O(1/k)
- Strongly convex: O(exp(-k/κ)), κ = L/μ (condition number)

**Momentum Methods:**

1. **Classical Momentum:**
   ```
   v_{k+1} = βv_k - α∇f(x_k)
   x_{k+1} = x_k + v_{k+1}
   ```

2. **Nesterov Accelerated Gradient:**
   ```
   y_{k+1} = x_k + β(x_k - x_{k-1})
   x_{k+1} = y_{k+1} - α∇f(y_{k+1})
   ```
   Rate: O(1/k²) for convex functions

3. **Adam (Adaptive Moment Estimation):**
   ```
   m_k = β₁m_{k-1} + (1-β₁)g_k
   v_k = β₂v_{k-1} + (1-β₂)g_k²
   m̂_k = m_k/(1-β₁ᵏ)
   v̂_k = v_k/(1-β₂ᵏ)
   x_{k+1} = x_k - α·m̂_k/√(v̂_k + ε)
   ```

### 3.4 Stochastic Optimization

**Stochastic Gradient Descent (SGD):**
```
x_{k+1} = x_k - α_k∇f_i(x_k)
where i ~ Uniform{1,...,n}
```

**Mini-batch SGD:**
- Batch size B: trade-off between variance and computation
- Learning rate schedules: constant, decay, cyclical

**Variance Reduction:**

1. **SVRG (Stochastic Variance Reduced Gradient):**
   - Maintains snapshot parameter x̃
   - Update: x_{k+1} = x_k - α[∇f_i(x_k) - ∇f_i(x̃) + ∇f(x̃)]

2. **SAGA:**
   - Stores gradients for all data points
   - Better memory-time tradeoff

### 3.5 Second-Order Methods

**Newton's Method:**
```
x_{k+1} = x_k - [∇²f(x_k)]⁻¹∇f(x_k)
```

**Quasi-Newton Methods:**

1. **BFGS:**
   ```
   B_{k+1} = B_k - (B_ks_k)(B_ks_k)ᵀ/(s_kᵀB_ks_k) + (y_ky_kᵀ)/(y_kᵀs_k)
   where s_k = x_{k+1} - x_k, y_k = ∇f(x_{k+1}) - ∇f(x_k)
   ```

2. **L-BFGS:**
   - Limited memory version
   - Stores only m recent update vectors

**Natural Gradient:**
```
x_{k+1} = x_k - αF⁻¹(x_k)∇f(x_k)
where F is the Fisher information matrix
```

### 3.6 Proximal Methods

**Proximal Operator:**
```
prox_f(v) = argmin_x [f(x) + (1/2)‖x-v‖²]
```

**Proximal Gradient Method:**
```
x_{k+1} = prox_{αg}(x_k - α∇f(x_k))
for f(x) + g(x) where g is non-smooth
```

**Applications:**
- ℓ₁ regularization (soft thresholding)
- Nuclear norm (SVD + thresholding)
- Projection onto constraints

**ADMM (Alternating Direction Method of Multipliers):**
```
minimize f(x) + g(z)
subject to Ax + Bz = c

Updates:
x_{k+1} = argmin_x [f(x) + (ρ/2)‖Ax+Bz_k-c+u_k‖²]
z_{k+1} = argmin_z [g(z) + (ρ/2)‖Ax_{k+1}+Bz-c+u_k‖²]
u_{k+1} = u_k + Ax_{k+1} + Bz_{k+1} - c
```

---

## 4. Information Theory

### 4.1 Entropy and Mutual Information

**Shannon Entropy:**
```
H(X) = -∑_x p(x)log p(x) = E[-log p(X)]
```

**Properties:**
- H(X) ≥ 0
- H(X) ≤ log|𝒳| (uniform maximizes entropy)
- Concave in p

**Conditional Entropy:**
```
H(Y|X) = E_X[H(Y|X=x)] = H(X,Y) - H(X)
```

**Mutual Information:**
```
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
       = D_KL(p(x,y) ‖ p(x)p(y))
```

**Chain Rule:**
- H(X₁,...,Xₙ) = ∑ᵢH(Xᵢ|X₁,...,Xᵢ₋₁)
- I(X₁,...,Xₙ;Y) = ∑ᵢI(Xᵢ;Y|X₁,...,Xᵢ₋₁)

### 4.2 Divergences

**Kullback-Leibler Divergence:**
```
D_KL(P‖Q) = E_P[log(P/Q)] = ∑_x p(x)log(p(x)/q(x))
```

**Properties:**
- D_KL(P‖Q) ≥ 0 (Gibbs' inequality)
- D_KL(P‖Q) = 0 ⟺ P = Q
- NOT symmetric, NOT a metric

**f-Divergences:**
```
D_f(P‖Q) = E_Q[f(dP/dQ)]
```

**Examples:**
- KL: f(t) = t log t
- Reverse KL: f(t) = -log t
- Total Variation: f(t) = |t-1|/2
- χ²: f(t) = (t-1)²
- Jensen-Shannon: symmetric version of KL

**Wasserstein Distance:**
```
W_p(μ,ν) = (inf_{γ∈Γ(μ,ν)} E_{(x,y)~γ}[‖x-y‖ᵖ])^(1/p)
```
- Used in: Optimal transport, GANs

### 4.3 Rate-Distortion Theory

**Rate-Distortion Function:**
```
R(D) = min_{p(x̂|x): E[d(x,x̂)]≤D} I(X;X̂)
```

**Applications:**
- Lossy compression
- Variational autoencoders
- Information bottleneck

### 4.4 Information Bottleneck

**Objective:**
```
min I(X;T) subject to I(T;Y) ≥ I_min
or
min I(X;T) - βI(T;Y)
```

**Applications:**
- Deep learning theory
- Feature selection
- Representation learning

---

## 5. Statistical Learning Theory

### 5.1 PAC Learning

**Probably Approximately Correct (PAC) Framework:**

A hypothesis class ℋ is PAC-learnable if there exists an algorithm A such that:
```
∀ε,δ > 0, ∀D: P(R(h) - R* ≤ ε) ≥ 1-δ
with sample complexity m ≤ poly(1/ε, 1/δ, |features|)
```

**Realizability Assumption:**
- ∃h* ∈ ℋ: R(h*) = 0

**Agnostic PAC Learning:**
- No realizability assumption
- Compete with best in class

### 5.2 VC Dimension

**Shattering:**
- ℋ shatters S if ∀b: S → {0,1}, ∃h ∈ ℋ consistent with b

**VC Dimension:**
```
VCdim(ℋ) = max{|S| : ℋ shatters S}
```

**Examples:**
- Linear classifiers in ℝᵈ: VCdim = d+1
- Neural networks: O(W log W) where W = # weights

**Fundamental Theorem of PAC Learning:**
```
ℋ is PAC-learnable ⟺ VCdim(ℋ) < ∞

Sample complexity: m = O((d/ε)log(1/δ))
where d = VCdim(ℋ)
```

### 5.3 Rademacher Complexity

**Empirical Rademacher Complexity:**
```
R̂_S(ℋ) = E_σ[sup_{h∈ℋ} (1/n)∑ᵢσᵢh(xᵢ)]
where σᵢ ~ Uniform{-1,+1}
```

**Rademacher Complexity:**
```
R_n(ℋ) = E_S[R̂_S(ℋ)]
```

**Generalization Bound:**
```
R(h) ≤ R̂(h) + 2R_n(ℋ) + √(log(1/δ)/(2n))
with probability ≥ 1-δ
```

### 5.4 Regularization and Structural Risk Minimization

**Empirical Risk Minimization (ERM):**
```
ĥ = argmin_{h∈ℋ} R̂(h)
```

**Regularized ERM:**
```
ĥ = argmin_{h∈ℋ} [R̂(h) + λΩ(h)]
```

**Common Regularizers:**
- ℓ₂ (Ridge, weight decay): Ω(w) = ‖w‖₂²
- ℓ₁ (Lasso, sparsity): Ω(w) = ‖w‖₁
- Elastic Net: Ω(w) = α‖w‖₁ + (1-α)‖w‖₂²
- Nuclear norm (low-rank): Ω(W) = ‖W‖_*

**Structural Risk Minimization:**
```
ℋ₁ ⊂ ℋ₂ ⊂ ... ⊂ ℋₙ
Choose ℋₖ balancing training error and complexity
```

### 5.5 Bias-Variance Decomposition

**For Squared Loss:**
```
E[(y - ĥ(x))²] = Bias²(ĥ(x)) + Var(ĥ(x)) + σ²

where:
Bias(ĥ(x)) = E[ĥ(x)] - h*(x)
Var(ĥ(x)) = E[(ĥ(x) - E[ĥ(x)])²]
σ² = irreducible error
```

**Trade-off:**
- Simple models: high bias, low variance
- Complex models: low bias, high variance
- Sweet spot: minimizes total error

### 5.6 Kernel Methods

**Kernel Function:**
```
k: 𝒳 × 𝒳 → ℝ
k(x,y) = ⟨φ(x), φ(y)⟩_ℋ
```

**Reproducing Kernel Hilbert Space (RKHS):**

**Properties:**
- Reproducing property: ⟨f, k(x,·)⟩ = f(x)
- Representer theorem

**Common Kernels:**

1. **Linear:** k(x,y) = xᵀy

2. **Polynomial:** k(x,y) = (xᵀy + c)ᵈ

3. **RBF (Gaussian):** k(x,y) = exp(-‖x-y‖²/(2σ²))

4. **Laplacian:** k(x,y) = exp(-‖x-y‖/σ)

**Representer Theorem:**
```
For f* = argmin_{f∈ℋ} [L(f) + λ‖f‖²_ℋ]
f* has form: f*(x) = ∑ᵢαᵢk(xᵢ,x)
```

**Kernel Ridge Regression:**
```
α = (K + λI)⁻¹y
where K_ij = k(xᵢ,xⱼ)
```

---

## 6. Deep Learning Mathematics

### 6.1 Neural Network Architectures

**Multilayer Perceptron:**
```
h^(ℓ+1) = σ(W^(ℓ)h^(ℓ) + b^(ℓ))
```

**Universal Approximation Theorem:**
- Single hidden layer with sufficient width can approximate any continuous function
- Width required: exponential in dimension (curse of dimensionality)
- Depth helps: hierarchical representations

**Activation Functions:**

1. **Sigmoid:** σ(x) = 1/(1+e⁻ˣ)
   - Range: (0,1)
   - Problem: vanishing gradients

2. **Tanh:** tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)
   - Range: (-1,1)
   - Zero-centered

3. **ReLU:** σ(x) = max(0,x)
   - Pros: no vanishing gradient, sparse activation
   - Cons: dying ReLU

4. **Leaky ReLU:** σ(x) = max(αx,x), α ∈ (0,1)

5. **ELU:** σ(x) = x if x>0, α(eˣ-1) otherwise

6. **GELU:** σ(x) = x·Φ(x) where Φ is Gaussian CDF
   - Used in Transformers

**Swish/SiLU:** σ(x) = x·sigmoid(βx)

### 6.2 Backpropagation

**Forward Pass:**
```
z^(ℓ) = W^(ℓ)a^(ℓ-1) + b^(ℓ)
a^(ℓ) = σ(z^(ℓ))
```

**Backward Pass (Chain Rule):**
```
δ^(L) = ∇_a L ⊙ σ'(z^(L))
δ^(ℓ) = ((W^(ℓ+1))ᵀδ^(ℓ+1)) ⊙ σ'(z^(ℓ))

∂L/∂W^(ℓ) = δ^(ℓ)(a^(ℓ-1))ᵀ
∂L/∂b^(ℓ) = δ^(ℓ)
```

**Computational Graph:**
- DAG representation
- Automatic differentiation
- Forward vs reverse mode

### 6.3 Normalization Techniques

**Batch Normalization:**
```
μ_B = (1/m)∑ᵢx_i
σ²_B = (1/m)∑ᵢ(x_i - μ_B)²
x̂_i = (x_i - μ_B)/√(σ²_B + ε)
y_i = γx̂_i + β
```

**Benefits:**
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization

**Layer Normalization:**
- Normalize across features (not batch)
- Used in: RNNs, Transformers

**Group Normalization:**
- Divide channels into groups
- Normalize within each group

**Weight Normalization:**
```
w = (g/‖v‖)v
where g is scalar, v is direction
```

### 6.4 Convolutional Neural Networks

**Convolution Operation:**
```
(f * g)(x) = ∑_y f(y)g(x-y)
```

**Discrete 2D Convolution:**
```
(I * K)(i,j) = ∑_m∑_n I(m,n)K(i-m,j-n)
```

**Cross-Correlation (actually used):**
```
(I * K)(i,j) = ∑_m∑_n I(i+m,j+n)K(m,n)
```

**Parameters:**
- Kernel size: k × k
- Stride: s
- Padding: p
- Output size: ((n+2p-k)/s)+1

**Receptive Field:**
- Region in input affecting output neuron
- Grows with depth
- RF_ℓ = RF_{ℓ-1} + (k-1)∏ᵢ₌₁^{ℓ-1}s_i

**Transposed Convolution:**
- Upsampling operation
- Used in: decoders, GANs

### 6.5 Recurrent Neural Networks

**Standard RNN:**
```
h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b_h)
y_t = W_hy·h_t + b_y
```

**Problems:**
- Vanishing/exploding gradients
- Limited long-term memory

**LSTM (Long Short-Term Memory):**
```
f_t = σ(W_f·[h_{t-1},x_t] + b_f)  (forget gate)
i_t = σ(W_i·[h_{t-1},x_t] + b_i)  (input gate)
C̃_t = tanh(W_C·[h_{t-1},x_t] + b_C)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t  (cell state)
o_t = σ(W_o·[h_{t-1},x_t] + b_o)  (output gate)
h_t = o_t ⊙ tanh(C_t)
```

**GRU (Gated Recurrent Unit):**
```
z_t = σ(W_z·[h_{t-1},x_t])  (update gate)
r_t = σ(W_r·[h_{t-1},x_t])  (reset gate)
h̃_t = tanh(W·[r_t⊙h_{t-1},x_t])
h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t
```

### 6.6 Attention Mechanisms

**Additive Attention (Bahdanau):**
```
e_ij = vᵀtanh(W_1h_i + W_2s_j)
α_ij = exp(e_ij)/∑_k exp(e_kj)
c_j = ∑_i α_ij h_i
```

**Multiplicative Attention (Luong):**
```
e_ij = h_iᵀW s_j
or e_ij = h_iᵀs_j  (dot-product)
```

**Scaled Dot-Product Attention:**
```
Attention(Q,K,V) = softmax(QKᵀ/√d_k)V
```

**Multi-Head Attention:**
```
MultiHead(Q,K,V) = Concat(head₁,...,head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Self-Attention:**
- Q, K, V all from same source
- Captures dependencies within sequence

### 6.7 Transformers

**Architecture:**
```
Encoder: Multi-Head Attention → Add&Norm → FFN → Add&Norm
Decoder: Masked Multi-Head Attention → Add&Norm →
         Cross Attention → Add&Norm → FFN → Add&Norm
```

**Positional Encoding:**
```
PE(pos,2i) = sin(pos/10000^(2i/d))
PE(pos,2i+1) = cos(pos/10000^(2i/d))
```

**Feed-Forward Network:**
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
or FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

**Complexity:**
- Self-attention: O(n²d) where n=sequence length
- Memory: O(n²) for attention weights

**Variants:**
- BERT: bidirectional encoder
- GPT: autoregressive decoder
- T5: encoder-decoder
- Vision Transformer (ViT)

### 6.8 Loss Functions

**Classification:**

1. **Cross-Entropy:**
   ```
   L = -∑_i y_i log ŷ_i
   ```

2. **Binary Cross-Entropy:**
   ```
   L = -[y log ŷ + (1-y)log(1-ŷ)]
   ```

3. **Focal Loss:**
   ```
   L = -α(1-p_t)^γ log p_t
   ```
   Addresses class imbalance

**Regression:**

1. **MSE:** L = (1/n)∑(y-ŷ)²

2. **MAE:** L = (1/n)∑|y-ŷ|

3. **Huber Loss:**
   ```
   L = ½(y-ŷ)²         if |y-ŷ| ≤ δ
       δ|y-ŷ| - ½δ²    otherwise
   ```

**Ranking:**
- Triplet Loss: L = max(0, d(a,p) - d(a,n) + margin)
- Contrastive Loss
- N-pair Loss

### 6.9 Generative Models

**Variational Autoencoders (VAE):**

**Objective (ELBO):**
```
log p(x) ≥ E_{z~q}[log p(x|z)] - D_KL(q(z|x)‖p(z))

Encoder: q(z|x) = N(μ(x), Σ(x))
Decoder: p(x|z)
Prior: p(z) = N(0,I)
```

**Reparameterization Trick:**
```
z = μ + σ ⊙ ε, where ε ~ N(0,I)
```

**Generative Adversarial Networks (GAN):**

**Minimax Objective:**
```
min_G max_D E_x[log D(x)] + E_z[log(1-D(G(z)))]

Generator: G: z → x
Discriminator: D: x → [0,1]
```

**Variants:**
- DCGAN: deep convolutional
- WGAN: Wasserstein distance
- StyleGAN: style-based
- Conditional GAN

**Diffusion Models:**

**Forward Process:**
```
q(x_t|x_{t-1}) = N(√(1-β_t)x_{t-1}, β_tI)
```

**Reverse Process:**
```
p_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))
```

**Training:** Predict noise ε_θ(x_t,t) ≈ ε

**Sampling:** Start from x_T ~ N(0,I), iteratively denoise

---

## 7. Advanced Topics

### 7.1 Graph Neural Networks

**Message Passing:**
```
h_v^(ℓ+1) = UPDATE(h_v^(ℓ), AGGREGATE({h_u^(ℓ) : u ∈ N(v)}))
```

**GCN (Graph Convolutional Network):**
```
H^(ℓ+1) = σ(D̃^{-½}ÃD̃^{-½}H^(ℓ)W^(ℓ))
where Ã = A + I
```

**GAT (Graph Attention Network):**
```
α_ij = softmax(e_ij)
e_ij = LeakyReLU(aᵀ[W h_i ‖ W h_j])
h_i' = σ(∑_{j∈N(i)} α_ij W h_j)
```

**GraphSAGE:**
- Sampling-based aggregation
- Inductive learning

### 7.2 Meta-Learning

**MAML (Model-Agnostic Meta-Learning):**

**Objective:**
```
min_θ ∑_{T_i} L_{T_i}(θ - α∇_θL_{T_i}(θ))
```

**Algorithm:**
1. Sample tasks T_i
2. Inner loop: θ_i' = θ - α∇_θL_{T_i}(θ)
3. Outer loop: θ ← θ - β∇_θ∑L_{T_i}(θ_i')

**Prototypical Networks:**
```
c_k = (1/|S_k|)∑_{x_i∈S_k} f_θ(x_i)
p(y=k|x) ∝ exp(-d(f_θ(x),c_k))
```

### 7.3 Reinforcement Learning Theory

**Markov Decision Process (MDP):**
- States: S
- Actions: A
- Transition: P(s'|s,a)
- Reward: R(s,a)
- Discount: γ ∈ [0,1)

**Value Functions:**
```
V^π(s) = E[∑_{t=0}^∞ γ^t R_t | s_0=s, π]
Q^π(s,a) = E[∑_{t=0}^∞ γ^t R_t | s_0=s, a_0=a, π]
```

**Bellman Equations:**
```
V^π(s) = ∑_a π(a|s)∑_{s'}P(s'|s,a)[R(s,a) + γV^π(s')]
Q^π(s,a) = ∑_{s'}P(s'|s,a)[R(s,a) + γ∑_{a'}π(a'|s')Q^π(s',a')]
```

**Optimal Policy:**
```
V*(s) = max_a Q*(s,a)
Q*(s,a) = ∑_{s'}P(s'|s,a)[R(s,a) + γV*(s')]
```

**Policy Gradient Theorem:**
```
∇_θJ(θ) = E[∑_t ∇_θlog π_θ(a_t|s_t)Q^π(s_t,a_t)]
```

### 7.4 Optimal Transport

**Kantorovich Problem:**
```
W_c(μ,ν) = inf_{γ∈Π(μ,ν)} ∫c(x,y)dγ(x,y)
```

**Sinkhorn Algorithm:**
- Entropic regularization
- Fast approximation
- Applications: domain adaptation, GANs

### 7.5 Neural Tangent Kernel

**Infinite Width Limit:**
```
f(x;θ) → GP as width → ∞
Kernel: Θ(x,x') = E[∂f(x)/∂θ · ∂f(x')/∂θ]
```

**Training Dynamics:**
```
df/dt = -η·Θ(x,x')∇f L
Linear in parameters at initialization
```

### 7.6 Causal Inference

**Structural Causal Models:**
```
X := f_X(PA_X, U_X)
```

**Do-Calculus:**
- Intervention: P(Y|do(X=x))
- Backdoor criterion
- Front-door criterion

**Causal Discovery:**
- PC algorithm
- GES (Greedy Equivalence Search)
- LiNGAM

---

## References and Further Reading

**Books:**
1. Boyd & Vandenberghe - Convex Optimization
2. Shalev-Shwartz & Ben-David - Understanding Machine Learning
3. Goodfellow, Bengio, Courville - Deep Learning
4. Bishop - Pattern Recognition and Machine Learning
5. MacKay - Information Theory, Inference, and Learning Algorithms

**Papers:**
- Attention Is All You Need (Vaswani et al., 2017)
- Deep Residual Learning (He et al., 2016)
- Adam: A Method for Stochastic Optimization (Kingma & Ba, 2015)
- Understanding the difficulty of training deep feedforward neural networks (Glorot & Bengio, 2010)

**Online Courses:**
- Stanford CS229 (Machine Learning)
- Stanford CS224N (NLP with Deep Learning)
- Stanford CS231N (Computer Vision)
- MIT 18.065 (Matrix Methods in Data Analysis)

---

## 8. Evaluation Metrics

### Why Metrics Matter

**Problem**: Loss functions optimize training, but don't directly measure what we care about

**Examples**:
- Low cross-entropy ≠ good user experience
- High accuracy with class imbalance ≠ useful model
- Small MSE ≠ perceptually good images

**Solution**: Use task-appropriate evaluation metrics

---

### 8.1 Classification Metrics

#### Binary Classification

**Confusion Matrix**:

```
                  Predicted
              Positive  Negative
Actual  Pos      TP        FN
        Neg      FP        TN
```

Where:
- **TP** (True Positive): Correctly predicted positive
- **TN** (True Negative): Correctly predicted negative
- **FP** (False Positive): Incorrectly predicted positive (Type I error)
- **FN** (False Negative): Incorrectly predicted negative (Type II error)

#### Basic Metrics

**1. Accuracy**

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

- Simple, intuitive
- **Problem**: Misleading with class imbalance
  - Example: 99% negative class → predict all negative → 99% accuracy!

**2. Precision** (Positive Predictive Value)

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

- "Of all predicted positives, how many were actually positive?"
- **High precision**: Few false alarms
- **Use when**: False positives are costly (e.g., spam detection)

**3. Recall** (Sensitivity, True Positive Rate)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

- "Of all actual positives, how many did we find?"
- **High recall**: Don't miss positives
- **Use when**: False negatives are costly (e.g., disease detection)

**4. F1 Score**

Harmonic mean of precision and recall:

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
$$

**Why harmonic mean?** Penalizes extreme imbalance

- Regular mean: $\frac{0.9 + 0.1}{2} = 0.5$
- Harmonic mean: $\frac{2 \cdot 0.9 \cdot 0.1}{0.9 + 0.1} = 0.18$

**Generalization**: $F_\beta$ score
$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

- $\beta > 1$: Favor recall
- $\beta < 1$: Favor precision

**5. Specificity** (True Negative Rate)

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

- "Of all actual negatives, how many did we correctly identify?"

#### ROC Curve and AUC

**ROC (Receiver Operating Characteristic) Curve**:
- Plot TPR (Recall) vs FPR at different thresholds
- $\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$

**AUC (Area Under Curve)**:
- Single number summary of ROC
- Range: [0, 1]
- 0.5: Random classifier
- 1.0: Perfect classifier

**Interpretation**: Probability that a random positive example is ranked higher than a random negative example

**Mathematical formulation**:
$$
\text{AUC} = \int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))
$$

#### Precision-Recall Curve

Alternative to ROC, especially for **imbalanced datasets**:
- Plot Precision vs Recall at different thresholds
- **AP (Average Precision)**: Area under PR curve

**When to use**:
- **Balanced classes**: ROC-AUC
- **Imbalanced classes**: Precision-Recall & AP

#### Multi-Class Classification

**1. Macro-Average**:
$$
\text{Macro-F1} = \frac{1}{C} \sum_{i=1}^C F1_i
$$

- Treat all classes equally
- Good when all classes are important

**2. Micro-Average**:
$$
\text{Micro-F1} = \frac{2 \sum_i TP_i}{2\sum_i TP_i + \sum_i FP_i + \sum_i FN_i}
$$

- Aggregate contributions, then compute
- Favors larger classes

**3. Weighted Average**:
$$
\text{Weighted-F1} = \sum_{i=1}^C \frac{n_i}{n} F1_i
$$

- Weight by class frequency

**Example**:
- Class A: 900 samples, F1=0.95
- Class B: 100 samples, F1=0.50

- Macro: $(0.95 + 0.50)/2 = 0.725$
- Micro: $\approx 0.92$ (dominated by class A)
- Weighted: $0.9 \times 0.95 + 0.1 \times 0.50 = 0.905$

#### Log Loss (Cross-Entropy)

For probabilistic predictions:

$$
\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]
$$

- Penalizes confident wrong predictions heavily
- **Lower is better**
- Measures calibration, not just decisions

---

### 8.2 Regression Metrics

#### 1. Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|
$$

- **Robust to outliers**
- Interpretable (same units as target)
- All errors weighted equally

#### 2. Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

- **Penalizes large errors** more heavily
- Differentiable everywhere
- Not robust to outliers

#### 3. Root Mean Squared Error (RMSE)

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$

- Same units as target
- Balances MSE properties with interpretability

#### 4. R² (Coefficient of Determination)

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
$$

Where $\bar{y} = \frac{1}{n}\sum_i y_i$ is the mean.

**Interpretation**:
- $R^2 = 1$: Perfect predictions
- $R^2 = 0$: Same as predicting mean
- $R^2 < 0$: Worse than predicting mean

**Problem**: Can be misleading
- Always increases with more features (for training data)
- Use adjusted $R^2$ or validation set

#### 5. Mean Absolute Percentage Error (MAPE)

$$
\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

- Scale-independent
- Interpretable as percentage
- **Problem**: Undefined when $y_i = 0$, asymmetric

---

### 8.3 Ranking and Recommendation Metrics

#### Precision@K

$$
\text{P@K} = \frac{\text{# relevant items in top K}}{K}
$$

Example: Search results, show top 10
- If 7 are relevant → P@10 = 0.7

#### Recall@K

$$
\text{R@K} = \frac{\text{# relevant items in top K}}{\text{total # relevant items}}
$$

#### Mean Average Precision (MAP)

$$
\text{MAP} = \frac{1}{|Q|}\sum_{q=1}^{|Q|} \text{AP}(q)
$$

Where for query $q$:
$$
\text{AP}(q) = \frac{1}{|R_q|}\sum_{k=1}^n P(k) \cdot \text{rel}(k)
$$

- $R_q$: Relevant documents for query $q$
- $P(k)$: Precision at position $k$
- $\text{rel}(k)$: 1 if item at $k$ is relevant, 0 otherwise

**Captures**: Precision across all recall levels

#### Normalized Discounted Cumulative Gain (NDCG)

For graded relevance (not just binary):

**DCG@K**:
$$
\text{DCG@K} = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i+1)}
$$

**NDCG@K**:
$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

Where IDCG = DCG of ideal ranking

- Range: [0, 1]
- 1 = perfect ranking
- Emphasizes top results (logarithmic discount)

#### Mean Reciprocal Rank (MRR)

$$
\text{MRR} = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

Where $\text{rank}_i$ is position of first relevant result for query $i$

**Use case**: Question answering, single correct answer

---

### 8.4 Generative Model Metrics

#### Perceptual Quality (Images)

**1. Inception Score (IS)**

$$
\text{IS} = \exp\left(\mathbb{E}_{x \sim p_g} D_{KL}(p(y|x) || p(y))\right)
$$

- Higher is better
- Measures diversity and quality
- **Problem**: Doesn't compare to real data distribution

**2. Fréchet Inception Distance (FID)**

$$
\text{FID} = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

Where:
- $(\mu_r, \Sigma_r)$: Mean and covariance of real images (in Inception feature space)
- $(\mu_g, \Sigma_g)$: Mean and covariance of generated images

- **Lower is better**
- Captures both quality and diversity
- Standard metric for GANs, diffusion models

#### Text Generation

**1. Perplexity**

$$
\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log p(w_i | w_{<i})\right)
$$

- Exponential of cross-entropy
- **Lower is better**
- Measures how "surprised" model is by test data

**2. BLEU (Bilingual Evaluation Understudy)**

$$
\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)
$$

Where:
- $p_n$: Precision of n-grams
- $BP$: Brevity penalty (penalize short outputs)

- Range: [0, 1]
- Measures overlap with reference translations
- **Problem**: Poor correlation with human judgment for single sentences

**3. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**

Focus on **recall** instead of precision:
- ROUGE-N: N-gram recall
- ROUGE-L: Longest common subsequence
- Used for summarization

**4. BERTScore**

$$
\text{BERTScore} = \frac{1}{|r|}\sum_{r_i \in r} \max_{c_j \in c} \text{cos}(r_i, c_j)
$$

- Semantic similarity using BERT embeddings
- Better than n-gram metrics for paraphrases

---

### 8.5 Clustering Metrics

#### Internal Metrics (No Ground Truth)

**1. Silhouette Score**

For each point $i$:
$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

Where:
- $a(i)$: Mean distance to other points in same cluster
- $b(i)$: Mean distance to points in nearest other cluster

**Average silhouette**: $\frac{1}{n}\sum_i s(i)$
- Range: [-1, 1]
- Higher is better

**2. Davies-Bouldin Index**

$$
\text{DB} = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}
$$

- **Lower is better**
- Ratio of within-cluster to between-cluster distances

#### External Metrics (With Ground Truth)

**1. Adjusted Rand Index (ARI)**

$$
\text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}
$$

- Range: [-1, 1]
- 1 = perfect match
- 0 = random clustering
- Adjusted for chance

**2. Normalized Mutual Information (NMI)**

$$
\text{NMI} = \frac{I(U; V)}{\sqrt{H(U)H(V)}}
$$

Where $U$ and $V$ are cluster assignments and ground truth

- Range: [0, 1]
- 1 = perfect clustering

---

### 8.6 Choosing the Right Metric

#### Decision Framework

```
Is this classification?
│
├─ Binary
│  ├─ Balanced classes? → Accuracy, F1
│  ├─ Imbalanced? → Precision-Recall, AP
│  ├─ Probabilistic? → Log Loss, ROC-AUC
│  └─ Cost-sensitive? → Custom threshold + F_β
│
├─ Multi-class
│  ├─ All classes equal? → Macro-F1
│  ├─ Size matters? → Weighted-F1
│  └─ Aggregate performance? → Micro-F1
│
Is this regression?
│  ├─ Outliers present? → MAE
│  ├─ Penalize large errors? → MSE, RMSE
│  ├─ Relative errors? → MAPE
│  └─ Variance explained? → R²
│
Is this ranking?
│  ├─ Top-K matters most? → P@K, R@K
│  ├─ Overall ranking? → MAP, NDCG
│  └─ First result critical? → MRR
│
Is this generation?
│  ├─ Images? → FID, IS
│  ├─ Text (reference)? → BLEU, ROUGE
│  ├─ Text (no reference)? → Perplexity
│  └─ Semantic quality? → BERTScore
│
Is this clustering?
│  ├─ No labels? → Silhouette, Davies-Bouldin
│  └─ With labels? → ARI, NMI
```

#### Best Practices

1. **Use multiple metrics**: No single metric is perfect
   - Example: Accuracy + Precision + Recall + ROC-AUC

2. **Match metric to business goal**:
   - Medical diagnosis: High recall (catch all diseases)
   - Spam filter: High precision (don't block legitimate email)

3. **Be aware of limitations**:
   - Accuracy fails with imbalance
   - BLEU doesn't capture fluency
   - FID requires many samples

4. **Report confidence intervals**:
   - Bootstrap resampling
   - Cross-validation
   - Statistical significance tests

5. **Consider computational cost**:
   - FID: Expensive (requires Inception network)
   - Accuracy: Cheap
   - Trade-off for development vs. final evaluation

---

### 8.7 Implementation Examples

#### Classification Metrics

```python
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

# Binary classification
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
y_prob = np.array([0.9, 0.1, 0.8, 0.6, 0.2, 0.95, 0.7, 0.3])

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
```

#### Regression Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = np.array([3.0, 2.5, 4.0, 7.0])
y_pred = np.array([2.8, 2.6, 3.8, 6.5])

print(f"MAE:  {mean_absolute_error(y_true, y_pred):.3f}")
print(f"MSE:  {mean_squared_error(y_true, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
print(f"R²:   {r2_score(y_true, y_pred):.3f}")
```

#### Custom Metrics

```python
def fbeta_score(precision, recall, beta=1.0):
    """Compute F-beta score."""
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

def mean_reciprocal_rank(ranks):
    """Compute MRR from list of ranks of first relevant item."""
    return np.mean(1.0 / np.array(ranks))

# Example
precision = 0.8
recall = 0.6

print(f"F1 (β=1):   {fbeta_score(precision, recall, beta=1.0):.3f}")
print(f"F2 (β=2):   {fbeta_score(precision, recall, beta=2.0):.3f}")  # Favor recall
print(f"F0.5 (β=0.5): {fbeta_score(precision, recall, beta=0.5):.3f}")  # Favor precision

# MRR example: first relevant at positions [1, 3, 2, 1]
ranks = [1, 3, 2, 1]
print(f"\nMRR: {mean_reciprocal_rank(ranks):.3f}")
```

---

### 8.8 Common Pitfalls

1. **Optimizing the wrong metric**
   - Training loss ≠ Evaluation metric
   - Solution: Use validation set with task metric

2. **Data leakage in metrics**
   - Computing metrics on training data
   - Including validation data in preprocessing
   - Solution: Strict train/val/test split

3. **Ignoring class imbalance**
   - High accuracy on imbalanced data
   - Solution: Use precision, recall, F1, or sampling

4. **Threshold selection**
   - Default 0.5 may not be optimal
   - Solution: Tune threshold on validation set

5. **Multiple comparisons**
   - Testing many models increases false discovery
   - Solution: Bonferroni correction, hold-out test set

6. **Cherry-picking metrics**
   - Reporting only favorable metrics
   - Solution: Pre-register evaluation protocol

---

## Practice Problems

### Linear Algebra

1. Prove that the SVD always exists for any m×n matrix.
2. Show that PCA is equivalent to finding the top eigenvectors of the covariance matrix.
3. Derive the gradient of the matrix determinant: ∂log|X|/∂X = X^{-T}

### Optimization

1. Prove convergence of gradient descent for strongly convex functions.
2. Derive the dual problem for SVM.
3. Implement ADMM for ℓ₁-regularized regression.

### Probability

1. Derive the variational lower bound (ELBO) for VAE.
2. Prove the Cramér-Rao lower bound.
3. Show that maximum likelihood is equivalent to minimizing KL divergence.

### Deep Learning

1. Derive backpropagation equations for a 3-layer network.
2. Prove the universal approximation theorem for single-layer networks.
3. Analyze the gradient flow through a ResNet block.

### Information Theory

1. Prove that mutual information is symmetric: I(X;Y) = I(Y;X)
2. Show that entropy is maximized by the uniform distribution.
3. Derive the information bottleneck objective.

---

*This guide provides graduate-level mathematical foundations for AI. Regular practice with these concepts through both theoretical problems and practical implementation is essential for mastery.*
