# Advanced Topics in AI Mathematics (2024 Edition)

*Cutting-edge mathematical foundations for modern AI systems*

---

## Table of Contents

### Part I: Large Language Models
1. [Transformer Architecture Deep Dive](#1-transformer-architecture-deep-dive)
2. [Attention Mechanisms](#2-attention-mechanisms)
3. [Positional Encodings](#3-positional-encodings)
4. [Scaling Laws](#4-scaling-laws)

### Part II: Diffusion Models
5. [Score-Based Generative Models](#5-score-based-generative-models)
6. [Denoising Diffusion](#6-denoising-diffusion)

### Part III: Efficient AI
7. [Model Compression](#7-model-compression)
8. [Quantization](#8-quantization)

---

## Part I: Large Language Models

## 1. Transformer Architecture Deep Dive

### 1.1 Scaled Dot-Product Attention

**Mathematical Definition**:

For queries $Q \in \mathbb{R}^{n \times d_k}$, keys $K \in \mathbb{R}^{m \times d_k}$, and values $V \in \mathbb{R}^{m \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Breakdown**:

1. **Similarity scores**: $S = QK^T \in \mathbb{R}^{n \times m}$
   - Each $S_{ij} = q_i^T k_j$ measures similarity between query $i$ and key $j$

2. **Scaling**: $\frac{S}{\sqrt{d_k}}$
   - **Why?** Prevents softmax saturation for large $d_k$
   - Variance of dot product grows with $d_k$: $\text{Var}(q^T k) = d_k$ (for unit variance $q, k$)
   - Scaling maintains variance ≈ 1

3. **Softmax**: $A = \text{softmax}\left(\frac{S}{\sqrt{d_k}}\right)$
   - Converts scores to probabilities: $A_{ij} \in [0,1]$, $\sum_j A_{ij} = 1$

4. **Weighted sum**: $\text{Output} = AV$
   - Each output is weighted combination of values

**Computational Complexity**:
- **Time**: $O(n \cdot m \cdot d_k + n \cdot m \cdot d_v)$
  - Dominant term: $O(n \cdot m \cdot d)$ where $d = d_k \approx d_v$
  - For self-attention ($n = m$): $O(n^2 d)$ **← Quadratic bottleneck!**
- **Memory**: $O(n \cdot m)$ for attention matrix
  - For long sequences, this is the limiting factor

**Why scaling by $\sqrt{d_k}$ matters**:

Without scaling:
```python
# For large d_k, dot products have large magnitude
Q, K = torch.randn(batch, seq_len, d_k), torch.randn(batch, seq_len, d_k)
scores = Q @ K.transpose(-2, -1)  # Large values!
attention = softmax(scores)  # Softmax saturates, gradients vanish

# With scaling:
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # Controlled variance
attention = softmax(scores)  # Better gradient flow
```

**Proof of variance growth**:

Assume $q, k \sim \mathcal{N}(0, I_{d_k})$. Then:

$$\begin{aligned}
\mathbb{E}[q^T k] &= \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = 0 \\
\text{Var}(q^T k) &= \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k
\end{aligned}$$

So scaling by $1/\sqrt{d_k}$ gives:
$$\text{Var}\left(\frac{q^T k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

### 1.2 Multi-Head Attention

**Intuition**: Attend to different representation subspaces in parallel

**Mathematical Formulation**:

$$\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{where } \text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}$$

**Parameter matrices**:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$

**Typical hyperparameters** (GPT-style):
- $h = 12$ to $96$ heads (increases with model size)
- $d_k = d_v = d_{\text{model}} / h$
- Example: GPT-3 has $d_{\text{model}} = 12288$, $h = 96$, so $d_k = 128$

**Why multiple heads?**

1. **Different relationships**: Each head can learn different patterns
   - Head 1: Syntax (subject-verb agreement)
   - Head 2: Semantics (word meaning relationships)
   - Head 3: Long-range dependencies

2. **Ensemble effect**: Reduces risk of poor attention patterns

3. **Computational efficiency**:
   - Could use single head with $d_k = d_{\text{model}}$
   - But splitting into $h$ smaller heads:
     - Same total parameters
     - More expressive (non-linear interaction via concat + linear)
     - Better parallelization

**Complexity**:
- **Per head**: $O(n^2 d_k)$
- **All heads**: $O(h n^2 d_k) = O(n^2 d_{\text{model}})$ (since $h \cdot d_k = d_{\text{model}}$)
- **Plus projections**: $O(n d_{\text{model}}^2)$
- **Total**: $O(n^2 d_{\text{model}} + n d_{\text{model}}^2)$
  - For $n > d_{\text{model}}$: dominated by $O(n^2 d_{\text{model}})$
  - For $n < d_{\text{model}}$: dominated by $O(n d_{\text{model}}^2)$

### 1.3 Self-Attention vs Cross-Attention

**Self-Attention**:
```
Q, K, V all come from the same source
Example: Encoder self-attention
  Input: sequence of tokens
  Output: contextual representations
```

**Cross-Attention**:
```
Q from one source, K and V from another
Example: Decoder cross-attention
  Q: from decoder (what we're generating)
  K, V: from encoder (input context)
```

**Masked Self-Attention** (for autoregressive models):
```python
# Prevent attention to future tokens
mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular
scores = scores.masked_fill(mask == 0, float('-inf'))
attention = softmax(scores)  # Future positions get 0 weight
```

**Causal mask** mathematical formulation:

$$\text{Attention}(Q, K, V)_{ij} = \begin{cases}
\text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right)_j V_j & \text{if } j \leq i \\
0 & \text{if } j > i
\end{cases}$$

### 1.4 Position-Wise Feed-Forward Networks

**Definition**:
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Or with GELU (used in GPT/BERT):
$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

**Dimensions**:
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$
- Typically: $d_{ff} = 4 \times d_{\text{model}}$ (GPT-2/3)

**Purpose**:
- Add non-linearity
- Increase model capacity
- Applied position-wise (same transformation at each position)

**Complexity**: $O(n d_{\text{model}} d_{ff}) = O(n d_{\text{model}}^2)$ (for $d_{ff} = 4d_{\text{model}}$)

### 1.5 Layer Normalization

**Pre-LN** (modern, used in GPT-3):
```python
def transformer_block_pre_ln(x):
    # Attention sub-layer
    x = x + attention(layer_norm(x))

    # FFN sub-layer
    x = x + ffn(layer_norm(x))

    return x
```

**Post-LN** (original Transformer):
```python
def transformer_block_post_ln(x):
    # Attention sub-layer
    x = layer_norm(x + attention(x))

    # FFN sub-layer
    x = layer_norm(x + ffn(x))

    return x
```

**Why Pre-LN is better**:
- More stable training for deep networks
- Gradients flow more directly through residual connections
- Less sensitive to learning rate

**LayerNorm mathematics**:

For input $x \in \mathbb{R}^d$:

$$\begin{aligned}
\mu &= \frac{1}{d}\sum_{i=1}^d x_i \\
\sigma^2 &= \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2 \\
\hat{x}_i &= \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
y_i &= \gamma \hat{x}_i + \beta
\end{aligned}$$

where $\gamma, \beta$ are learned parameters.

**Difference from BatchNorm**:
- BatchNorm: normalize across batch dimension
- LayerNorm: normalize across feature dimension
- LayerNorm works better for sequences (no batch statistics needed)

---

## 2. Attention Mechanisms

### 2.1 Attention Variants

#### 2.1.1 Linear Attention

**Problem**: Standard attention is $O(n^2)$ in sequence length

**Idea**: Approximate softmax attention with linear complexity

**Kernel trick**:
$$\text{Attention}(Q, K, V) = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})}$$

where $\phi$ is a feature map such that:
$$\phi(q)^T \phi(k) \approx \exp\left(\frac{q^T k}{\sqrt{d_k}}\right)$$

**Complexity**: $O(n d^2)$ instead of $O(n^2 d)$
- For $n \gg d$: huge speedup!

**Trade-off**: Approximation quality

#### 2.1.2 Flash Attention

**Problem**: Standard attention is memory-bound

**Key insight**: Fuse operations to reduce memory reads/writes

**Algorithm** (simplified):
1. Tile Q, K, V into blocks
2. Compute attention for each block
3. Accumulate results on-chip (SRAM)
4. Never materialize full attention matrix in global memory (HBM)

**Benefits**:
- Same result as standard attention (mathematically equivalent)
- 2-4x faster in practice
- Enables longer sequences

**Complexity**:
- **Time**: Still $O(n^2 d)$ FLOPs
- **Memory**: $O(n d)$ instead of $O(n^2 + nd)$
- **I/O**: $O(n^2 d^2 / M)$ instead of $O(n^2 d)$ (where $M$ = SRAM size)

#### 2.1.3 Sparse Attention

**Idea**: Only attend to subset of positions

**Patterns**:

1. **Local attention**: Attend to window of size $w$
   $$A_{ij} \neq 0 \text{ only if } |i - j| \leq w$$
   Complexity: $O(nwd)$

2. **Strided attention**: Attend to every $k$-th position
   $$A_{ij} \neq 0 \text{ only if } j \equiv 0 \pmod{k}$$

3. **Block sparse**: Combination of local + strided
   - Used in BigBird, Longformer

**Trade-off**:
- ✅ Faster for long sequences
- ❌ May miss some long-range dependencies

### 2.2 Grouped Query Attention (GQA)

**Motivation**: Reduce KV cache size for inference

**Standard Multi-Head Attention**:
- $h$ separate $Q, K, V$ projections
- $h$ separate KV caches
- Memory: $O(2 \cdot h \cdot n \cdot d_k)$ for KV cache

**Multi-Query Attention (MQA)**:
- $h$ separate $Q$ projections
- **1 shared** $K, V$ projection
- Memory: $O(2 \cdot n \cdot d_k)$ (independent of $h$)
- Trade-off: Lower quality

**Grouped Query Attention (GQA)** [compromise]:
- Group heads into $g$ groups ($1 < g < h$)
- Each group shares $K, V$
- Example: $h=32$, $g=8$ → 4 heads share each KV
- Memory: $O(2 \cdot g \cdot n \cdot d_k)$

**Used in**: Llama 2, Mistral

---

## 3. Positional Encodings

### 3.1 Absolute Positional Encoding

**Original Transformer** (Vaswani et al., 2017):

$$\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\end{aligned}$$

**Properties**:
- Fixed (not learned)
- Allows model to attend to relative positions
- Can extrapolate to longer sequences than seen in training

**Why sinusoidal?**

For any offset $k$:
$$PE_{pos+k} = f(PE_{pos})$$
for some linear function $f$. This allows the model to learn relative positions.

**Proof**:
$$\begin{aligned}
\sin(\alpha + \beta) &= \sin\alpha\cos\beta + \cos\alpha\sin\beta \\
\cos(\alpha + \beta) &= \cos\alpha\cos\beta - \sin\alpha\sin\beta
\end{aligned}$$

So $PE_{pos+k}$ is a linear combination of $PE_{pos}$ and $PE_k$.

### 3.2 Learned Positional Embeddings

**Alternative**: Learn position embeddings like word embeddings

**Pros**:
- Can adapt to task
- Often performs better on training length

**Cons**:
- Doesn't extrapolate to longer sequences
- Requires more parameters

**Used in**: BERT, GPT-2 (early versions)

### 3.3 Rotary Position Embedding (RoPE)

**Idea** (Su et al., 2021): Encode position by rotating query/key vectors

**Formulation**:

For position $m$, rotate query and key by angle $m\theta$:

$$\begin{aligned}
q_m &= R_\Theta^m q \\
k_m &= R_\Theta^m k
\end{aligned}$$

where $R_\Theta^m$ is rotation matrix.

**Key property**: Attention score depends only on relative position:
$$q_m^T k_n = q^T (R_\Theta^m)^T R_\Theta^n k = q^T R_\Theta^{n-m} k$$

**Advantages**:
- Naturally encodes relative positions
- Extrapolates to longer sequences
- No extra parameters

**Implementation** (for 2D subspace):
$$\begin{bmatrix}
q_{2i} \\ q_{2i+1}
\end{bmatrix} \leftarrow \begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix} \begin{bmatrix}
q_{2i} \\ q_{2i+1}
\end{bmatrix}$$

**Used in**: GPT-Neo, GPT-J, Llama, many modern LLMs

### 3.4 ALiBi (Attention with Linear Biases)

**Idea** (Press et al., 2021): Add position-dependent bias to attention scores

**Formulation**:
$$\text{softmax}(Q_i K^T / \sqrt{d_k} + m \cdot [-i, \ldots, -2, -1, 0])$$

where $m$ is a head-specific slope (learned or fixed).

**Example**:
- Head 1: $m = -0.5$
- Head 2: $m = -1.0$
- Position biases: $[0, -1, -2, -3, \ldots]$ (closer = less penalty)

**Advantages**:
- Simple to implement
- Excellent extrapolation to longer sequences
- No extra parameters per position
- Faster than learned embeddings

**Used in**: BLOOM, MPT

---

## 4. Scaling Laws

### 4.1 Neural Scaling Laws (Kaplan et al., 2020)

**Empirical finding**: Loss scales as power law with compute, data, parameters

**Formulas**:
$$\begin{aligned}
L(N) &\sim N^{-\alpha} & \text{(parameters)} \\
L(D) &\sim D^{-\beta} & \text{(data)} \\
L(C) &\sim C^{-\gamma} & \text{(compute)}
\end{aligned}$$

where $\alpha \approx 0.076$, $\beta \approx 0.095$, $\gamma \approx 0.050$ for language models.

**Combined**:
$$L(N, D) = \left(\frac{N_c}{N}\right)^\alpha + \left(\frac{D_c}{D}\right)^\beta + L_\infty$$

### 4.2 Chinchilla Scaling Laws (Hoffmann et al., 2022)

**Finding**: Previous models were **under-trained**!

**Optimal allocation**:
For compute budget $C$:
- Optimal $N \propto C^{0.5}$
- Optimal $D \propto C^{0.5}$

**Implication**: Should scale model size AND data proportionally

**Example**:
- GPT-3: 175B params, 300B tokens → under-trained
- Chinchilla: 70B params, 1.4T tokens → better with less params!

**Practical rule of thumb**:
$$D \approx 20 \times N$$
(tokens ≈ 20× parameters)

### 4.3 Emergence

**Definition**: Capabilities that appear suddenly at scale

**Examples**:
- In-context learning
- Chain-of-thought reasoning
- Multi-step reasoning

**Mathematical characterization** (work in progress):
- Phase transitions in loss landscape?
- Grokking phenomena?

---

## 5. Computational Optimizations

### 5.1 KV Cache

**Problem**: Autoregressive generation recomputes past keys and values

**Solution**: Cache K and V for all past positions

**Memory**:
- Per layer: $2 \times \text{batch} \times \text{seq\_len} \times d_{\text{model}}$
- For $L$ layers: $2L \times \text{batch} \times \text{seq\_len} \times d_{\text{model}}$

**Example** (Llama 2 70B):
- $L = 80$, $d_{\text{model}} = 8192$, FP16
- 1 token per sequence: $2 \times 80 \times 8192 \times 2 = 2.6$ MB
- 2048 tokens: $2.6 \times 2048 = 5.3$ GB per sequence!

**This is why**:
- Batch size is limited during generation
- Grouped Query Attention (GQA) helps
- Quantization helps (KV cache in INT8)

### 5.2 PagedAttention (vLLM)

**Idea**: Manage KV cache like virtual memory

**Benefits**:
- Reduce memory fragmentation
- Share KV cache across sequences (for beam search)
- Higher throughput

---

## Quick Reference: Key Equations

### Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Multi-Head:
  head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
  MultiHead = Concat(head_1, ..., head_h)W^O
```

### Transformer Block (Pre-LN)
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### RoPE
```
q_m = R(m θ) q
k_n = R(n θ) k
q_m^T k_n depends only on (n - m)
```

### Scaling Laws
```
Loss ~ N^(-0.076)  (parameters)
Loss ~ D^(-0.095)  (data)

Chinchilla: D ≈ 20N
```

---

## Part II: Diffusion Models

## 6. Score-Based Generative Models

### 6.1 Foundation: Score Matching

**Goal**: Learn the gradient of the log probability density (the "score"):

$$s_\theta(x) \approx \nabla_x \log p(x)$$

**Why?** Can generate samples without computing $p(x)$ explicitly!

**Score matching objective** (Hyvärinen, 2005):
$$L(\theta) = \frac{1}{2}\mathbb{E}_{p(x)}\left[\|s_\theta(x) - \nabla_x \log p(x)\|^2\right]$$

**Problem**: Don't know $\nabla_x \log p(x)$ (that's what we're trying to learn!)

**Solution - Denoising Score Matching**:

Add noise: $\tilde{x} = x + \sigma \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$

$$L_{\text{DSM}}(\theta) = \mathbb{E}_{p(x), p(\epsilon)}\left[\left\|s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log p(\tilde{x}|x)\right\|^2\right]$$

Since $p(\tilde{x}|x) = \mathcal{N}(x, \sigma^2 I)$:
$$\nabla_{\tilde{x}} \log p(\tilde{x}|x) = -\frac{\tilde{x} - x}{\sigma^2} = -\frac{\epsilon}{\sigma}$$

**Final objective** (equivalent):
$$L_{\text{DSM}}(\theta) = \mathbb{E}_{x, \epsilon}\left[\left\|s_\theta(x + \sigma\epsilon) + \frac{\epsilon}{\sigma}\right\|^2\right]$$

### 6.2 Langevin Dynamics Sampling

**Given** score function $s_\theta(x) \approx \nabla_x \log p(x)$, **generate** samples:

$$x_{t+1} = x_t + \frac{\eta}{2} s_\theta(x_t) + \sqrt{\eta} z_t$$

where $z_t \sim \mathcal{N}(0, I)$, $\eta$ is step size.

**Intuition**: Gradient ascent on log probability + noise

**Result**: $x_T \sim p(x)$ as $T \to \infty$, $\eta \to 0$

---

## 7. Denoising Diffusion Probabilistic Models (DDPM)

### 7.1 Forward Process (Diffusion)

**Add noise gradually** over $T$ steps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

where $\beta_1, \ldots, \beta_T$ is **variance schedule** (e.g., $\beta_t \in [0.0001, 0.02]$).

**Closed form** for any $t$:
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$$

where:
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$

**Reparameterization**:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Property**: As $t \to T$, $x_T \approx \mathcal{N}(0, I)$ (pure noise)

### 7.2 Reverse Process (Denoising)

**Goal**: Learn to reverse the diffusion

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**Key insight** (from score matching):

The reverse process mean should be:
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

where $\epsilon_\theta$ is a neural network predicting the noise.

**Variance**: Often fixed to $\Sigma_\theta = \beta_t I$ or $\tilde{\beta}_t I$ where:
$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$

### 7.3 Training Objective

**Variational Lower Bound**:
$$L = \mathbb{E}_q\left[-\log p_\theta(x_0|x_1) + \sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))\right]$$

**Simplified** (Ho et al., 2020):
$$L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

where:
- $t \sim \text{Uniform}\{1, \ldots, T\}$
- $x_0 \sim q(x_0)$ (real data)
- $\epsilon \sim \mathcal{N}(0, I)$
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$

**Algorithm** (Training):
```python
1. Sample x_0 from data
2. Sample t ~ Uniform{1, ..., T}
3. Sample ε ~ N(0, I)
4. Compute x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
5. Compute loss: ||ε - ε_θ(x_t, t)||²
6. Update θ via gradient descent
```

**Algorithm** (Sampling):
```python
1. Sample x_T ~ N(0, I)
2. For t = T, ..., 1:
     ε ~ N(0, I) if t > 1, else ε = 0
     x_{t-1} = 1/√α_t (x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t, t)) + √β_t ε
3. Return x_0
```

### 7.4 Mathematical Derivations

**Derivation 1: Forward process closed form**

Want: $q(x_t | x_0)$

By induction:
$$\begin{aligned}
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1} \\
    &= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}) + \sqrt{1-\alpha_t}\epsilon_{t-1} \\
    &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1}
\end{aligned}$$

Using $\mathcal{N}(0, \sigma_1^2) + \mathcal{N}(0, \sigma_2^2) = \mathcal{N}(0, \sigma_1^2 + \sigma_2^2)$:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

**Derivation 2: Reverse process mean**

Posterior: $q(x_{t-1} | x_t, x_0)$ is Gaussian with mean:
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$$

Substitute $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\epsilon)$:
$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)$$

Since we don't know $\epsilon$, we predict it: $\epsilon_\theta(x_t, t)$.

### 7.5 Connection to Score-Based Models

**Score function**:
$$\nabla_{x_t} \log q(x_t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\epsilon$$

**Predicted score**:
$$s_\theta(x_t, t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)$$

**Therefore**:
- Predicting noise $\epsilon$ ≡ Predicting score function
- DDPM is a discretization of score-based SDE

---

## 8. Advanced Diffusion Techniques

### 8.1 Variance Schedules

**Linear** (original DDPM):
$$\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)$$

**Cosine** (improved, Nichol & Dhariwal, 2021):
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

**Why cosine?** Slower noise addition at start/end, more uniform SNR.

### 8.2 Improved Sampling: DDIM

**Problem**: DDPM requires T steps (slow!)

**DDIM** (Song et al., 2020): Non-Markovian, deterministic sampling

**Update rule**:
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\epsilon_\theta(x_t) + \sigma_t \epsilon_t$$

**Special case** $\sigma_t = 0$: **Deterministic**!
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t)}{\sqrt{\bar{\alpha}_t}}}_{\text{predicted } x_0} + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t)$$

**Benefit**: Can skip steps! Use $t \in \{1, 10, 20, \ldots, T\}$ instead of $\{1, 2, \ldots, T\}$
- 10x-50x faster sampling
- Slightly lower quality (acceptable trade-off)

### 8.3 Guidance

#### 8.3.1 Classifier Guidance

**Goal**: Conditional generation $p(x|y)$

**Bayes' rule**:
$$\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)$$

**Guided sampling**:
$$\tilde{\epsilon}_\theta(x_t) = \epsilon_\theta(x_t) - \sqrt{1-\bar{\alpha}_t}\nabla_{x_t} \log p(y|x_t)$$

**Requires**: Pre-trained classifier $p(y|x_t)$ (noisy classifier!)

#### 8.3.2 Classifier-Free Guidance (CFG)

**Idea**: Train conditional and unconditional models together

**Training**:
- With probability $p$: condition on label $y$
- With probability $1-p$: drop label (unconditional)

**Single model** learns both:
- $\epsilon_\theta(x_t, y)$ (conditional)
- $\epsilon_\theta(x_t, \emptyset)$ (unconditional)

**Guided prediction**:
$$\tilde{\epsilon}_\theta(x_t, y) = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, y) - \epsilon_\theta(x_t, \emptyset))$$

where $w$ is **guidance scale** ($w=0$: unconditional, $w=1$: conditional, $w>1$: over-guided).

**Intuition**:
- Amplify the conditional signal
- Move away from unconditional prediction

**Used in**: Stable Diffusion, DALL-E 2, Imagen

**Typical values**: $w \in [5, 15]$ for images

---

## 9. Latent Diffusion Models

**Problem**: Diffusion in pixel space is expensive

**Idea**: Diffusion in **latent space**

### 9.1 Architecture

1. **Encoder**: $E: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{h \times w \times c}$
   - Compress image to latent (e.g., 8x8x4 from 512x512x3)

2. **Diffusion**: Operate in latent space
   - Much smaller: $h \ll H$, $w \ll W$

3. **Decoder**: $D: \mathbb{R}^{h \times w \times c} \to \mathbb{R}^{H \times W \times 3}$
   - Reconstruct image from latent

**Training**:
1. Train autoencoder (E, D) with perceptual loss
2. Freeze encoder/decoder
3. Train diffusion model in latent space

**Used in**: Stable Diffusion

**Benefits**:
- 3-8x faster training
- 3-8x faster sampling
- Same quality
- Lower memory

---

## 10. Quick Reference: Diffusion Equations

### DDPM Training
```
Loss = E_{t,x₀,ε} [||ε - ε_θ(x_t, t)||²]

where:
  x_t = √(ᾱ_t) x₀ + √(1-ᾱ_t) ε
  t ~ Uniform{1,...,T}
  ε ~ N(0, I)
```

### DDPM Sampling
```
x_T ~ N(0, I)
for t = T to 1:
  z ~ N(0, I) if t > 1, else 0
  x_{t-1} = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t,t)) + √β_t z
```

### DDIM Sampling (Deterministic)
```
x₀_pred = (x_t - √(1-ᾱ_t) ε_θ(x_t)) / √ᾱ_t
x_{t-1} = √ᾱ_{t-1} x₀_pred + √(1-ᾱ_{t-1}) ε_θ(x_t)
```

### Classifier-Free Guidance
```
ε̃ = ε_θ(x_t, ∅) + w(ε_θ(x_t, y) - ε_θ(x_t, ∅))

Common w: 7-15 for images
```

---

## References (Continued)

8. **Ho, J., Jain, A., & Abbeel, P.** (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.

9. **Song, J., Meng, C., & Ermon, S.** (2020). *Denoising Diffusion Implicit Models*. ICLR 2021.

10. **Dhariwal, P., & Nichol, A.** (2021). *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS.

11. **Ho, J., & Salimans, T.** (2022). *Classifier-Free Diffusion Guidance*. NeurIPS Workshop.

12. **Rombach, R., et al.** (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR. (Stable Diffusion)

13. **Song, Y., & Ermon, S.** (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS.

---

**Next**: Model Compression, Quantization (Phase 3)

---

## References

1. **Vaswani, A., et al.** (2017). *Attention Is All You Need*. NeurIPS.

2. **Su, J., et al.** (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv.

3. **Press, O., et al.** (2021). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*. ICLR.

4. **Kaplan, J., et al.** (2020). *Scaling Laws for Neural Language Models*. arXiv.

5. **Hoffmann, J., et al.** (2022). *Training Compute-Optimal Large Language Models*. arXiv. (Chinchilla paper)

6. **Dao, T., et al.** (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS.

7. **Ainslie, J., et al.** (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. arXiv.

---

# PART III: Model Compression and Efficiency

## 11. Introduction to Model Compression

### Motivation

Modern deep learning models are becoming increasingly large:
- **GPT-3**: 175B parameters (~700GB in FP32)
- **PaLM**: 540B parameters
- **Stable Diffusion**: 890M parameters (~3.4GB)

**Problems**:
1. **Memory**: Cannot fit on single GPU
2. **Latency**: Slow inference for real-time applications
3. **Energy**: High power consumption
4. **Cost**: Expensive cloud deployment

**Goal**: Reduce model size and computational cost while maintaining accuracy

### Compression Techniques Overview

| Technique | Compression Ratio | Accuracy Impact | Inference Speedup |
|-----------|------------------|----------------|-------------------|
| **Pruning** | 2-10× | Low (1-2%) | 2-3× |
| **Quantization** | 4× (INT8) | Very Low (<1%) | 2-4× |
| **Knowledge Distillation** | Variable | Low-Medium | Variable |
| **Low-Rank Factorization** | 2-5× | Low | 1.5-2× |
| **Neural Architecture Search** | Variable | Can improve | Variable |

---

## 12. Pruning

### 12.1 Magnitude-Based Pruning

**Idea**: Remove weights with small magnitudes

**Unstructured Pruning**: Remove individual weights
$$
W_{\text{pruned}}[i,j] = \begin{cases}
W[i,j] & \text{if } |W[i,j]| > \tau \\
0 & \text{otherwise}
\end{cases}
$$

**Choosing threshold** $\tau$:
- **Global**: Threshold across all layers
- **Layer-wise**: Different threshold per layer
- **Top-k%**: Keep only top k% largest weights

**Structured Pruning**: Remove entire neurons/channels/filters

$$
\text{Keep channel } c \text{ if } ||W_c|| > \tau
$$

Where $||W_c||$ is the L2 norm of all weights in channel $c$.

### 12.2 Iterative Pruning (Lottery Ticket Hypothesis)

**Algorithm**:
```
1. Train network to convergence
2. Prune p% smallest magnitude weights
3. Reset remaining weights to initial values
4. Retrain
5. Repeat steps 2-4
```

**Lottery Ticket Hypothesis**: A randomly initialized network contains a subnetwork ("winning ticket") that can match the original network's performance when trained in isolation.

### 12.3 Practical Implementation

```python
def magnitude_pruning(weight, sparsity=0.5):
    """
    Prune weights by magnitude.

    Args:
        weight: Weight tensor
        sparsity: Fraction of weights to prune (0.5 = 50% pruned)
    """
    # Compute threshold
    threshold = np.percentile(np.abs(weight), sparsity * 100)

    # Create mask
    mask = np.abs(weight) > threshold

    # Apply mask
    pruned_weight = weight * mask

    return pruned_weight, mask

# Example: Prune 90% of weights
W = np.random.randn(512, 512)
W_pruned, mask = magnitude_pruning(W, sparsity=0.9)

print(f"Original non-zero: {np.count_nonzero(W)}")
print(f"Pruned non-zero: {np.count_nonzero(W_pruned)}")
print(f"Sparsity: {1 - np.count_nonzero(W_pruned) / W.size:.1%}")
```

### 12.4 Structured Pruning: Channel Pruning

```python
def channel_pruning(weight, sparsity=0.3):
    """
    Prune entire channels based on L2 norm.

    Args:
        weight: Conv weight tensor (out_channels, in_channels, H, W)
        sparsity: Fraction of channels to prune
    """
    # Compute L2 norm per output channel
    channel_norms = np.linalg.norm(weight.reshape(weight.shape[0], -1), axis=1)

    # Threshold
    threshold = np.percentile(channel_norms, sparsity * 100)

    # Mask: keep channels with norm > threshold
    mask = channel_norms > threshold

    # Apply mask
    pruned_weight = weight[mask, :, :, :]

    return pruned_weight, mask
```

---

## 13. Quantization

### 13.1 Basics of Quantization

**Goal**: Represent weights/activations with fewer bits

**Standard representations**:
- **FP32** (32-bit float): Standard training
- **FP16** (16-bit float): Mixed precision training
- **INT8** (8-bit integer): Common for inference
- **INT4** / **Binary**: Extreme compression

### 13.2 Uniform Quantization

Map floating point values to integers:

$$
q = \text{round}\left(\frac{x - z}{s}\right)
$$

$$
x \approx s \cdot q + z
$$

Where:
- $x$: Original float value
- $q$: Quantized integer value
- $s$: **Scale** (step size)
- $z$: **Zero-point** (offset)

**Symmetric quantization** ($z = 0$):
$$
s = \frac{\max(|x_{\min}|, |x_{\max}|)}{2^{b-1} - 1}
$$

**Asymmetric quantization**:
$$
s = \frac{x_{\max} - x_{\min}}{2^b - 1}, \quad z = -\text{round}\left(\frac{x_{\min}}{s}\right)
$$

Where $b$ is the number of bits.

### 13.3 Quantization-Aware Training (QAT)

**Problem**: Post-training quantization can lose accuracy

**Solution**: Simulate quantization during training

**Straight-Through Estimator (STE)**:
- **Forward**: Use quantized values
- **Backward**: Pretend quantization is identity function

$$
\frac{\partial \text{round}(x)}{\partial x} \approx 1
$$

```python
def quantize_aware_forward(x, scale, zero_point, num_bits=8):
    """
    Quantization-aware forward pass.
    """
    # Quantize
    q_min = 0
    q_max = 2**num_bits - 1

    q = np.round(x / scale + zero_point)
    q = np.clip(q, q_min, q_max)

    # Dequantize (for forward pass)
    x_quant = (q - zero_point) * scale

    return x_quant

# In backward pass: gradient flows through as if no quantization
```

### 13.4 Per-Channel vs Per-Tensor Quantization

**Per-Tensor**: Single scale for entire tensor
- Simple, fast
- Less accurate

**Per-Channel**: Different scale per output channel
- More accurate
- Slightly more complex

$$
q_{i,j} = \text{round}\left(\frac{W_{i,j}}{s_i}\right)
$$

Where $s_i$ is the scale for output channel $i$.

### 13.5 Practical INT8 Quantization

```python
def compute_quantization_params(x, num_bits=8):
    """
    Compute scale and zero-point for quantization.
    """
    q_min = 0
    q_max = 2**num_bits - 1

    x_min = x.min()
    x_max = x.max()

    # Asymmetric quantization
    scale = (x_max - x_min) / (q_max - q_min)
    zero_point = q_min - x_min / scale
    zero_point = np.round(zero_point).astype(np.int32)
    zero_point = np.clip(zero_point, q_min, q_max)

    return scale, zero_point

def quantize(x, scale, zero_point, num_bits=8):
    """Quantize float tensor to int."""
    q_min = 0
    q_max = 2**num_bits - 1

    q = np.round(x / scale + zero_point)
    q = np.clip(q, q_min, q_max).astype(np.uint8)

    return q

def dequantize(q, scale, zero_point):
    """Dequantize int tensor to float."""
    return (q.astype(np.float32) - zero_point) * scale

# Example
W = np.random.randn(512, 512) * 0.1
scale, zero_point = compute_quantization_params(W, num_bits=8)
W_quant = quantize(W, scale, zero_point)
W_dequant = dequantize(W_quant, scale, zero_point)

error = np.abs(W - W_dequant).mean()
print(f"Quantization error: {error:.6f}")
print(f"Compression: {W.nbytes / W_quant.nbytes:.1f}x")
```

---

## 14. Knowledge Distillation

### 14.1 Basic Knowledge Distillation

**Idea**: Train small "student" model to mimic large "teacher" model

**Setup**:
- **Teacher**: Large, accurate model (pre-trained)
- **Student**: Smaller model (to be trained)

**Loss Function**:

$$
\mathcal{L} = \alpha \mathcal{L}_{\text{CE}}(y, \hat{y}_s) + (1-\alpha) \mathcal{L}_{\text{KD}}(\hat{y}_t, \hat{y}_s)
$$

Where:
- $y$: True labels
- $\hat{y}_s$: Student predictions
- $\hat{y}_t$: Teacher predictions
- $\alpha$: Balance parameter (e.g., 0.5)

**Distillation Loss** (Hinton et al., 2015):

$$
\mathcal{L}_{\text{KD}} = \text{KL}\left(\text{softmax}\left(\frac{z_t}{T}\right) || \text{softmax}\left(\frac{z_s}{T}\right)\right)
$$

Where:
- $z_t, z_s$: Logits (before softmax)
- $T$: **Temperature** (typically 3-20)

**Why temperature?** Softens probability distribution, revealing more information about similarities between classes.

### 14.2 Temperature Scaling

**Hard targets** (T=1): $[0.98, 0.01, 0.01]$
- One class dominates
- Little information about relationships

**Soft targets** (T=5): $[0.60, 0.25, 0.15]$
- Reveals class similarities
- Richer training signal

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

### 14.3 Feature Distillation

Transfer **intermediate representations**, not just outputs:

$$
\mathcal{L}_{\text{feat}} = ||h_t - W \cdot h_s||^2
$$

Where:
- $h_t$: Teacher's intermediate features
- $h_s$: Student's intermediate features
- $W$: Projection matrix (if dimensions differ)

### 14.4 Self-Distillation

**Idea**: Use model's own predictions as soft targets

**Benefits**:
- Smooths training
- Acts as regularization
- Can improve performance even without compression

### 14.5 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
    """
    Knowledge distillation loss.

    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits (detached)
        labels: Ground truth labels
        temperature: Softmax temperature
        alpha: Balance between CE and KD loss
    """
    # Hard target loss (standard cross-entropy)
    hard_loss = F.cross_entropy(student_logits, labels)

    # Soft target loss (KL divergence with temperature scaling)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)

    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    soft_loss = soft_loss * (temperature ** 2)  # Scale by T^2

    # Combined loss
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

    return total_loss

# Example usage
teacher = TeacherModel()  # Large model
student = StudentModel()  # Small model

# Training loop
for inputs, labels in dataloader:
    # Teacher predictions (no gradient)
    with torch.no_grad():
        teacher_logits = teacher(inputs)

    # Student predictions
    student_logits = student(inputs)

    # Distillation loss
    loss = distillation_loss(student_logits, teacher_logits, labels,
                             temperature=5.0, alpha=0.3)

    # Backward and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 15. Low-Rank Factorization

### 15.1 Matrix Factorization

**Idea**: Approximate weight matrix $W \in \mathbb{R}^{m \times n}$ with low-rank decomposition

**SVD Decomposition**:
$$
W \approx U_k \Sigma_k V_k^T
$$

Where $k \ll \min(m, n)$ is the rank.

**Compression**:
- Original: $m \times n$ parameters
- Factorized: $k(m + n)$ parameters
- Ratio: $\frac{mn}{k(m+n)}$

**Example**: $W \in \mathbb{R}^{512 \times 512}$, $k=64$
- Original: 262,144 parameters
- Factorized: 65,536 parameters
- Compression: 4×

### 15.2 Tucker Decomposition for CNNs

**Convolutional layers**: $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K \times K}$

**Tucker decomposition**:
$$
W \approx G \times_1 U^{(1)} \times_2 U^{(2)} \times_3 U^{(3)} \times_4 U^{(4)}
$$

Approximate as sequence of smaller convolutions.

### 15.3 LoRA (Low-Rank Adaptation)

**Recent approach for fine-tuning** (Hu et al., 2021)

Instead of updating all weights, add low-rank updates:

$$
W' = W + BA
$$

Where:
- $W \in \mathbb{R}^{d \times k}$: Original (frozen)
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: Trainable low-rank matrices
- $r \ll \min(d, k)$: Rank (e.g., 8)

**Benefits**:
- Only train $r(d+k)$ parameters instead of $dk$
- Can switch between tasks by swapping $B, A$
- No inference overhead (merge $BA$ into $W$)

**Used in**: Stable Diffusion fine-tuning, LLM adaptation

```python
class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.
    """
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()

        # Frozen original weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # Low-rank factors (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.rank = rank

    def forward(self, x):
        # Original transformation
        result = F.linear(x, self.weight)

        # Add low-rank update
        result += F.linear(F.linear(x, self.lora_A), self.lora_B)

        return result
```

---

## 16. Practical Comparison

### 16.1 Compression Techniques Comparison

| Technique | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Pruning** | Simple, effective | Irregular sparsity hard to accelerate | Memory-constrained deployment |
| **Quantization** | Large speedup, HW support | Accuracy drop possible | Mobile, edge devices |
| **Distillation** | Flexible, can improve | Requires teacher training | Small models from scratch |
| **Low-Rank** | Mathematically principled | Limited compression | Large matrix multiplications |

### 16.2 Combining Techniques

**Common pipeline**:
1. **Distillation**: Train smaller architecture
2. **Quantization-Aware Training**: Add quantization
3. **Pruning**: Remove redundant weights
4. **Fine-tuning**: Recover accuracy

**Example**: MobileNetV2 → DistilBERT approach
- Architectural efficiency
- Knowledge transfer
- Quantization

### 16.3 Hardware Considerations

**GPU**:
- INT8 support: 2-4× speedup (Tensor Cores)
- Sparsity support: 2× speedup (Ampere+)

**CPU**:
- INT8: Intel VNNI, ARM dot product instructions
- Significant speedup for inference

**Mobile/Edge**:
- INT8 essential
- Model size critical (limited memory)

---

## 17. Tools and Frameworks

### PyTorch

```python
# Quantization
import torch.quantization as quant

# Post-training static quantization
model_fp32 = MyModel()
model_fp32.eval()
model_fp32.qconfig = quant.get_default_qconfig('fbgemm')
model_prepared = quant.prepare(model_fp32)
# Calibrate with data
model_int8 = quant.convert(model_prepared)

# Quantization-aware training
model_fp32.qconfig = quant.get_default_qat_qconfig('fbgemm')
model_prepared = quant.prepare_qat(model_fp32)
# Train
model_int8 = quant.convert(model_prepared)
```

### TensorFlow Lite

```python
# TensorFlow Lite quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # or tf.int8

tflite_model = converter.convert()
```

### ONNX Runtime

- Cross-framework optimization
- Quantization, graph optimization
- Hardware-specific acceleration

---

## 18. Case Studies

### 18.1 MobileNet

**Architecture optimizations**:
- Depthwise separable convolutions
- Inverted residuals
- Width/resolution multipliers

**Compression techniques**:
- Quantization (8-bit)
- Pruning

**Results**: 75% accuracy on ImageNet with <5M parameters

### 18.2 DistilBERT

**Approach**: Knowledge distillation from BERT

**Techniques**:
- 6 layers instead of 12
- Distillation on masked LM task
- Triple loss (MLM + distillation + cosine embedding)

**Results**:
- 40% smaller
- 60% faster
- 97% of BERT's performance

### 18.3 Stable Diffusion + LoRA

**Fine-tuning approach**:
- Freeze UNet weights
- Add LoRA layers (rank 4-16)
- Train on custom dataset

**Benefits**:
- <10MB per fine-tuned model
- Fast training (< 1 hour on single GPU)
- No catastrophic forgetting

---

## 19. Best Practices

### When to Use Each Technique

1. **Pruning**:
   - When model is over-parameterized
   - Memory is constrained
   - Can retrain/fine-tune

2. **Quantization**:
   - Production deployment
   - Real-time inference needed
   - Target hardware supports INT8

3. **Knowledge Distillation**:
   - Training smaller models from scratch
   - Have access to unlabeled data
   - Can train teacher model

4. **Low-Rank Factorization**:
   - Large linear layers
   - Fine-tuning large models
   - Need parameter efficiency

### Recommended Pipeline

**For Research**:
```
Original Model → Pruning → Fine-tuning → Evaluation
```

**For Production**:
```
Architecture Search → Distillation → QAT → Pruning → Deployment
```

### Common Pitfalls

1. **Quantizing too early**: Train in FP32, quantize at end
2. **Aggressive pruning**: Start with 30-50%, increase gradually
3. **Ignoring hardware**: Quantization speedup depends on hardware support
4. **Not fine-tuning**: Always fine-tune after compression

---

## 20. Future Directions

### Emerging Techniques

1. **Neural Architecture Search (NAS)**: Automated compression-aware design
2. **Mixed-Precision**: Automatic bit-width selection per layer
3. **Sparse Training**: Train sparse from initialization (no pruning phase)
4. **Quantization at Scale**: LLM.int8(), GPTQ for billion-parameter models

### Research Questions

- Theoretical understanding of why compression works
- Compression for multimodal models
- Hardware-software co-design
- Compression-aware training from scratch

---

## References (Part III)

14. **Han, S., et al.** (2015). *Learning both Weights and Connections for Efficient Neural Networks*. NeurIPS.

15. **Frankle, J., & Carbin, M.** (2019). *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*. ICLR.

16. **Jacob, B., et al.** (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR.

17. **Hinton, G., Vinyals, O., & Dean, J.** (2015). *Distilling the Knowledge in a Neural Network*. NeurIPS Workshop.

18. **Sanh, V., et al.** (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. NeurIPS Workshop.

19. **Hu, E. J., et al.** (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.

20. **Dettmers, T., et al.** (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS.

---

**End of Part III: Model Compression**
