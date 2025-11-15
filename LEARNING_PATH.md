# AI Mathematics Learning Path

*A structured guide to mastering AI mathematics from foundations to cutting-edge research*

---

## 📍 Overview

This document provides **structured learning paths** based on your:
- Current knowledge level
- Learning goals
- Available time
- Career objectives

**Estimated total time**: 3-6 months (10-15 hours/week)

---

## 🎯 Choose Your Path

### Path A: Complete Beginner → AI Researcher
**For**: Undergraduates, career changers with STEM background
**Duration**: 4-6 months
**Goal**: Master AI mathematics from scratch

### Path B: ML Engineer → Deep Learning Expert
**For**: Software engineers, data scientists
**Duration**: 2-3 months
**Goal**: Deepen mathematical understanding, implement modern architectures

### Path C: Graduate Student → Research Specialist
**For**: Master's/PhD students in AI/ML
**Duration**: 1-2 months
**Goal**: Advanced topics, paper implementation, research skills

---

## 📊 Self-Assessment

Before starting, complete [PREREQUISITES.md](./PREREQUISITES.md) checklist.

**Your Level**:
- **Level 0** (< 50% prerequisites): Follow Path A
- **Level 1** (50-70% prerequisites): Follow Path B
- **Level 2** (70-90% prerequisites): Follow Path C
- **Level 3** (> 90% prerequisites): Cherry-pick advanced topics

---

## 🛤️ Path A: Complete Beginner → AI Researcher

### Phase 0: Prerequisites (2-4 weeks)

**If math foundations are weak:**

1. **Linear Algebra** (1-2 weeks)
   - Resource: 3Blue1Brown - Essence of Linear Algebra
   - Resource: MIT 18.06 (lectures 1-16)
   - Practice: Khan Academy exercises

2. **Calculus** (1-2 weeks)
   - Focus: Multivariate calculus, gradients
   - Resource: Khan Academy Multivariable Calculus
   - Practice: Compute gradients by hand

3. **Probability** (1 week)
   - Focus: Distributions, expectation, Bayes' theorem
   - Resource: Harvard Stat 110 (first 10 lectures)

**Checkpoint**: Can you compute $\\nabla_x (x^T A x)$? Do you understand Bayes' theorem?

### Phase 1: Mathematical Foundations (4-6 weeks)

#### Week 1-2: Linear Algebra for AI

**Study**:
1. Read [AI_Mathematics_Guide.md](./AI_Mathematics_Guide.md) - Section 1
2. Work through [SVD_Deep_Dive.ipynb](./notebooks/01_linear_algebra/SVD_Deep_Dive.ipynb)

**Exercises**:
- Implement PCA from scratch
- Apply SVD to image compression
- Solve least squares problems

**Checkpoint**: Can you explain why SVD is used in PCA?

#### Week 3-4: Probability and Statistics

**Study**:
1. Read [AI_Mathematics_Guide.md](./AI_Mathematics_Guide.md) - Section 2
2. Interactive exercises in [index.html](./index.html) - Probability section

**Exercises**:
- Implement Naive Bayes classifier
- Compute MLE for Gaussian distribution
- Bayesian inference for coin flips

**Checkpoint**: Understand the difference between frequentist and Bayesian approaches?

#### Week 5-6: Optimization Theory

**Study**:
1. Read [AI_Mathematics_Guide.md](./AI_Mathematics_Guide.md) - Section 3
2. Interactive gradient descent in [index.html](./index.html)

**Exercises**:
- Implement gradient descent for linear regression
- Compare SGD, Momentum, Adam
- Solve simple convex optimization problems

**Checkpoint**: Can you derive the gradient descent update rule?

### Phase 2: Neural Networks (4-6 weeks)

#### Week 7-8: Backpropagation

**Study**:
1. Read [AI_Mathematics_Guide.md](./AI_Mathematics_Guide.md) - Section 6
2. Complete [Backpropagation_Mathematics.ipynb](./notebooks/04_neural_networks/Backpropagation_Mathematics.ipynb)

**Exercises**:
- Derive backprop equations for 2-layer network
- Implement neural network from scratch (no PyTorch/TF!)
- Gradient checking

**Checkpoint**: Can you implement and debug a simple neural network?

#### Week 9-10: Deep Learning Fundamentals

**Study**:
1. Numerical stability: [Numerical_Stability_Guide.md](./Numerical_Stability_Guide.md)
2. Activation functions, regularization, normalization

**Exercises**:
- Implement stable softmax
- Compare different activation functions
- Apply batch normalization

**Checkpoint**: Understand why we use log probabilities?

#### Week 11-12: Modern Architectures

**Study**:
1. CNNs, RNNs, Attention (from AI_Mathematics_Guide.md)
2. Transformers: [Advanced_Topics_2024.md](./Advanced_Topics_2024.md) - Part I

**Project**:
- Implement a simple image classifier (CNN)
- Implement sequence model (LSTM or GRU)
- Study Transformer architecture

**Checkpoint**: Can you explain self-attention mathematically?

### Phase 3: Advanced Topics (4-6 weeks)

#### Week 13-14: Large Language Models

**Study**:
1. Complete [Advanced_Topics_2024.md](./Advanced_Topics_2024.md) - Sections 1-5
2. Study scaling laws, position encodings

**Project**:
- Implement multi-head attention
- Study GPT/BERT architectures
- Analyze computational complexity

#### Week 15-16: Generative Models

**Study**:
1. [Advanced_Topics_2024.md](./Advanced_Topics_2024.md) - Sections 6-10 (Diffusion Models)
2. GANs, VAEs (from AI_Mathematics_Guide.md)

**Project**:
- Implement DDPM for MNIST
- Study Stable Diffusion architecture
- Compare VAE vs GAN vs Diffusion

#### Week 17-18: Research & Implementation

**Choose one**:
- Implement a recent paper (2023-2024)
- Reproduce benchmark results
- Extend existing work

**Resources**:
- Papers with Code
- Hugging Face
- OpenReview

---

## 🛤️ Path B: ML Engineer → Deep Learning Expert

### Prerequisites Check (1 week)

- ✅ Python & NumPy proficiency
- ✅ Basic ML algorithms (linear regression, logistic regression)
- ✅ Some deep learning experience (used PyTorch/TensorFlow)

**Start here**: Linear Algebra review

### Fast Track (8-10 weeks)

#### Week 1: Math Bootcamp

**Linear Algebra**:
- [SVD_Deep_Dive.ipynb](./notebooks/01_linear_algebra/SVD_Deep_Dive.ipynb) (focus on applications)
- PCA, matrix factorization

**Optimization**:
- Gradient descent variants
- Convexity basics

**Skip**: Detailed proofs, focus on intuition

#### Week 2-3: Neural Networks Deep Dive

**Study**:
1. [Backpropagation_Mathematics.ipynb](./notebooks/04_neural_networks/Backpropagation_Mathematics.ipynb)
2. [Numerical_Stability_Guide.md](./Numerical_Stability_Guide.md) (entire guide!)

**Practice**:
- Implement backprop from scratch
- Debug numerical issues in your own models
- Gradient checking

**Goal**: Truly understand what PyTorch is doing under the hood

#### Week 4-5: Transformers

**Study**:
1. [Advanced_Topics_2024.md](./Advanced_Topics_2024.md) - Sections 1-5
2. Attention mechanisms in detail

**Project**:
- Implement Transformer from scratch
- Fine-tune BERT/GPT for your task
- Analyze attention patterns

**Goal**: Master modern NLP architectures

#### Week 6-7: Generative Models

**Study**:
1. [Advanced_Topics_2024.md](./Advanced_Topics_2024.md) - Sections 6-10
2. Diffusion models

**Project**:
- Implement DDPM
- Fine-tune Stable Diffusion
- Generate custom images

**Goal**: Understand DALL-E, Midjourney, Stable Diffusion

#### Week 8-10: Practical Skills

**Topics**:
- Mixed precision training
- Model compression (quantization, pruning)
- Distributed training
- MLOps for large models

**Project**:
- Deploy a large model
- Optimize inference speed
- Benchmark different approaches

**Goal**: Production-ready deep learning

---

## 🛤️ Path C: Graduate Student → Research Specialist

### Assumed Knowledge
- Strong mathematical background
- Experience with research papers
- Proficiency in deep learning frameworks

### Accelerated Track (4-6 weeks)

#### Week 1: Mathematical Rigor

**Focus on proofs and derivations**:
1. All mathematical derivations in [AI_Mathematics_Guide.md](./AI_Mathematics_Guide.md)
2. Prove convergence of optimization algorithms
3. Statistical learning theory (PAC learning, VC dimension)

**Goal**: Prove theorems, not just use them

#### Week 2: Advanced Architectures

**Study**:
1. [Advanced_Topics_2024.md](./Advanced_Topics_2024.md) - Complete Part I
2. Recent papers on Transformers (2023-2024)
3. Scaling laws, emergence

**Project**:
- Implement Flash Attention
- Analyze complexity of different attention variants
- Reproduce scaling law experiments

#### Week 3: Generative Models

**Study**:
1. [Advanced_Topics_2024.md](./Advanced_Topics_2024.md) - Complete Part II
2. Score-based models theory
3. SDEs, probability flow ODEs

**Project**:
- Derive DDPM equations from scratch
- Implement guidance mechanisms
- Compare different samplers

#### Week 4-6: Research Project

**Choose a research direction**:
1. **Efficiency**: Sparse attention, low-rank adapters, quantization
2. **Theory**: Scaling laws, optimization landscapes, generalization
3. **Applications**: Domain-specific models, multimodal learning
4. **Safety**: Adversarial robustness, interpretability, alignment

**Deliverable**:
- Literature review
- Novel experiment or theoretical result
- Technical report or paper draft

---

## 📅 Study Schedules

### Full-Time (30+ hours/week)
- Complete Path A in 2-3 months
- Complete Path B in 1-1.5 months
- Complete Path C in 3-4 weeks

### Part-Time (10-15 hours/week)
- Complete Path A in 4-6 months
- Complete Path B in 2-3 months
- Complete Path C in 1.5-2 months

### Weekend Warrior (5-10 hours/week)
- Complete Path A in 6-9 months
- Complete Path B in 3-4 months
- Complete Path C in 2-3 months

---

## 🎓 Milestones & Assessments

### Milestone 1: Foundations (After Phase 1)
**Can you**:
- [ ] Derive gradient of matrix expressions?
- [ ] Implement gradient descent from scratch?
- [ ] Explain bias-variance tradeoff?
- [ ] Compute Bayesian posterior?

**If yes**: Proceed to neural networks
**If no**: Review weak areas

### Milestone 2: Neural Networks (After Phase 2)
**Can you**:
- [ ] Implement backprop for arbitrary architecture?
- [ ] Debug NaN/Inf in training?
- [ ] Explain why we use specific activations/normalizations?
- [ ] Use mixed precision training?

**If yes**: Proceed to advanced topics
**If no**: More practice with implementations

### Milestone 3: Advanced (After Phase 3)
**Can you**:
- [ ] Implement Transformer from scratch?
- [ ] Explain attention complexity trade-offs?
- [ ] Derive diffusion model equations?
- [ ] Read and understand recent papers?

**If yes**: Ready for research!
**If no**: Focus on specific weak areas

---

## 💡 Study Tips

### 1. Active Learning
❌ **Don't**: Just read passively
✅ **Do**: Derive equations yourself, implement algorithms

### 2. Interleave Theory and Practice
- Read theory → Implement → Debug → Deepen understanding
- Don't spend weeks on pure theory

### 3. Use Multiple Resources
- **This repository**: Core curriculum
- **Papers**: Latest research
- **Courses**: Structure and assignments
- **Textbooks**: Comprehensive coverage

### 4. Projects are Key
- Build something every week
- Start simple, increase complexity
- Debug errors = deep learning

### 5. Community
- Study groups
- Paper reading clubs
- Stack Overflow, Reddit r/MachineLearning
- Twitter/X for staying current

---

## 📚 Recommended Supplementary Resources

### Books
1. **Goodfellow et al.** - Deep Learning (comprehensive)
2. **Bishop** - Pattern Recognition and Machine Learning (Bayesian perspective)
3. **Boyd & Vandenberghe** - Convex Optimization (optimization theory)
4. **Murphy** - Probabilistic Machine Learning (modern, comprehensive)

### Online Courses
1. **Stanford CS229** - Machine Learning (Andrew Ng)
2. **Stanford CS224N** - NLP with Deep Learning
3. **Stanford CS231N** - Computer Vision
4. **Fast.ai** - Practical Deep Learning
5. **NYU DS-GA 1008** - Deep Learning (Yann LeCun)

### Papers to Read
1. **Attention Is All You Need** (Vaswani et al., 2017)
2. **BERT** (Devlin et al., 2018)
3. **GPT-3** (Brown et al., 2020)
4. **Denoising Diffusion Probabilistic Models** (Ho et al., 2020)
5. **FlashAttention** (Dao et al., 2022)

### Staying Current
- **Papers with Code**: https://paperswithcode.com/
- **arXiv Sanity**: http://www.arxiv-sanity.com/
- **Hugging Face Daily Papers**: https://huggingface.co/papers
- **Twitter/X**: Follow @karpathy, @ylecun, @soumithchintala, @_jasonwei

---

## 🏆 Career Paths After Completion

### Research Scientist
**Requirements**:
- PhD (usually)
- Strong publication record
- Novel research contributions

**Next steps**:
- Work on research projects
- Publish at top conferences (NeurIPS, ICML, ICLR, CVPR)
- PhD program or research lab

### ML Engineer
**Requirements**:
- Strong implementation skills
- Production ML experience
- System design knowledge

**Next steps**:
- Contribute to open source (PyTorch, Hugging Face)
- Build production systems
- Portfolio of deployed models

### Applied Scientist
**Requirements**:
- Master's or PhD
- Research + implementation skills
- Domain expertise

**Next steps**:
- Specialize in application area (NLP, CV, etc.)
- Bridge research and product
- Industry research labs

### AI Consultant
**Requirements**:
- Broad knowledge
- Communication skills
- Business acumen

**Next steps**:
- Diverse project portfolio
- Networking
- Thought leadership (blog, talks)

---

## 🆘 When You Get Stuck

### Common Issues

**"Math is too hard"**
→ Go back to prerequisites, use visual resources (3Blue1Brown)
→ Focus on intuition first, rigor later

**"Can't debug my code"**
→ Use gradient checking
→ Start with tiny examples (2x2 matrices)
→ Compare with reference implementations

**"Papers are incomprehensible"**
→ Start with blog posts/videos about the paper
→ Read "Introduction" and "Conclusion" first
→ Skip heavy math initially, come back later

**"Feeling overwhelmed"**
→ Take breaks
→ Focus on one concept at a time
→ Remember: everyone finds this challenging!

### Getting Help

1. **GitHub Issues**: Report errors in this repository
2. **Stack Overflow**: Technical questions
3. **r/MachineLearning**: Conceptual questions, discussions
4. **Discord/Slack**: Real-time help (various ML communities)
5. **Office Hours**: If in academic setting

---

## 📊 Track Your Progress

### Suggested Tracking Method

Create a spreadsheet:

| Week | Topic | Hours | Status | Notes |
|------|-------|-------|--------|-------|
| 1 | Linear Algebra | 12 | ✅ | SVD notebook complete |
| 2 | Probability | 10 | 🔄 | Bayesian inference challenging |
| ... | ... | ... | ... | ... |

### Weekly Review

Every Sunday:
1. What did I learn this week?
2. What was challenging?
3. What should I review?
4. What's next week's goal?

---

## 🎉 Congratulations!

If you've completed your chosen path, you now have:
- ✅ Deep mathematical understanding of AI
- ✅ Implementation skills for modern architectures
- ✅ Ability to read and understand research papers
- ✅ Foundation for research or advanced engineering

**Next**: Apply your knowledge, build projects, contribute to research!

---

## 📞 Feedback

This learning path is continuously evolving. Please:
- Report issues or unclear sections
- Suggest improvements
- Share your success stories
- Contribute additional resources

**Happy Learning! 🚀**

---

**Last Updated**: 2024
**Maintained by**: AI Mathematics Study Group
