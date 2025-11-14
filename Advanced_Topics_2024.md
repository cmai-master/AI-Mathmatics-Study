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

**To be continued**: Diffusion Models, Model Compression, and more...

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

**Next Section**: [Diffusion Models](#part-ii-diffusion-models) (Coming in next update)
