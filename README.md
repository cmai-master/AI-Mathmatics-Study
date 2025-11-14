# AI Mathematics Study Guide

석사급 깊이의 AI 수학 학습 가이드입니다.

## 📚 포함 내용

### 1. **학습 문서** (`AI_Mathematics_Guide.md`)
대학원 수준의 AI 수학 이론을 다루는 포괄적인 가이드:

- **Linear Algebra**: SVD, 고유값 분해, 행렬 미적분학, 텐서 연산
- **Probability Theory**: 측도론적 확률, 베이지안 통계, 집중 부등식
- **Optimization Theory**: 볼록 최적화, Gradient Descent, 근사 방법, ADMM
- **Information Theory**: 엔트로피, KL Divergence, Rate-Distortion 이론
- **Statistical Learning Theory**: PAC Learning, VC Dimension, Rademacher Complexity
- **Deep Learning Mathematics**: 역전파, 정규화 기법, Transformer 수학
- **Advanced Topics**: GNN, Meta-Learning, Reinforcement Learning, Optimal Transport

### 2. **Interactive Webpage** (`index.html`)
수학 개념을 시각화하고 실습할 수 있는 인터랙티브 웹페이지:

#### 주요 기능:

**🎯 인터랙티브 시각화:**
- SVD 분해 및 저계수 근사
- 고유값 변환 시각화
- 다양한 Norm(ℓ₁, ℓ₂, ℓ∞) 비교
- Gaussian 분포 파라미터 조정
- 베이지안 업데이트 실시간 시뮬레이션
- Central Limit Theorem 데모
- Gradient Descent 최적화 경로
- 활성화 함수 및 미분 비교
- Entropy 계산기
- KL Divergence 시각화
- Bias-Variance Tradeoff
- ℓ₁ vs ℓ₂ 정규화 비교

**💻 코드 실습:**
- Python 코드 에디터 (Pyodide 기반)
- Gradient Descent 구현 연습
- SVD 계산 및 응용
- NumPy를 활용한 실습

**📝 연습 문제:**
- 객관식 퀴즈 (즉각적인 피드백)
- 수학적 유도 문제
- 코딩 챌린지

**🔬 사용 기술:**
- KaTeX: 수학 공식 렌더링
- Plotly: 인터랙티브 그래프
- Pyodide: 브라우저 내 Python 실행
- Pure JavaScript: 모든 시각화 및 계산

## 🚀 사용 방법

### 학습 문서 읽기:
```bash
# Markdown 뷰어로 열기
cat AI_Mathematics_Guide.md

# 또는 GitHub에서 렌더링된 버전 보기
```

### Interactive Webpage 실행:
```bash
# 브라우저에서 index.html 파일 열기
open index.html

# 또는 로컬 서버 실행 (권장)
python -m http.server 8000
# 브라우저에서 http://localhost:8000 접속
```

**온라인 버전:** GitHub Pages에 배포하면 바로 사용 가능합니다.

## 📖 학습 경로

### 초급 → 중급 (기초 다지기)
1. Linear Algebra 기본 개념
2. Probability Theory 기초
3. Basic Optimization

### 중급 → 고급 (심화 학습)
4. Advanced Optimization (ADMM, Proximal Methods)
5. Information Theory
6. Statistical Learning Theory

### 고급 (연구 수준)
7. Deep Learning Mathematics
8. Advanced Topics (GNN, Meta-Learning, RL)

## 🎓 학습 목표

이 가이드를 완료하면:
- ✅ AI 논문의 수학적 내용을 이해할 수 있습니다
- ✅ 새로운 알고리즘을 수학적으로 분석할 수 있습니다
- ✅ 최적화 문제를 공식화하고 해결할 수 있습니다
- ✅ 학습 이론의 보장(guarantees)을 이해합니다
- ✅ 딥러닝 모델의 이론적 기초를 설명할 수 있습니다

## 🔗 참고 자료

**교과서:**
- Boyd & Vandenberghe - Convex Optimization
- Shalev-Shwartz & Ben-David - Understanding Machine Learning
- Goodfellow et al. - Deep Learning
- Bishop - Pattern Recognition and Machine Learning

**강의:**
- Stanford CS229 (Machine Learning)
- Stanford CS224N (NLP with Deep Learning)
- MIT 18.065 (Matrix Methods)

## 📊 난이도 수준

이 자료는 다음 배경지식을 가진 학생들을 위해 설계되었습니다:
- 학부 수준의 선형대수학
- 미적분학 및 다변수 미적분
- 기초 확률론 및 통계
- Python 프로그래밍

**목표 독자:** 석사 과정 학생, AI 연구자, ML 엔지니어

## 🛠️ 기술 스택

- **문서:** Markdown with LaTeX
- **웹:** HTML5, CSS3, JavaScript (ES6+)
- **수학 렌더링:** KaTeX
- **시각화:** Plotly.js
- **Python 실행:** Pyodide
- **코드 하이라이팅:** Highlight.js

## 📄 라이선스

Educational purposes - 자유롭게 학습 및 공유 가능

## 🤝 기여

이 프로젝트는 AI 수학 학습을 위한 오픈 리소스입니다. 개선사항이나 추가 내용이 있다면 기여해주세요!

---

**Happy Learning! 🎓📐🤖**
