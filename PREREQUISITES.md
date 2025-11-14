# Prerequisites - 사전 지식 체크리스트

이 AI Mathematics Study를 효과적으로 학습하기 위해 필요한 사전 지식을 확인하세요.

---

## 🎯 목표 학습자

- **대학원생** (석사 과정 이상)
- **AI/ML 연구자**
- **시니어 ML 엔지니어**
- **논문을 읽고 구현하고 싶은 개발자**

---

## ✅ 필수 사전 지식 (Must-Have)

### 1. 수학 기초

#### 1.1 선형대수학 (Linear Algebra)
**필수 개념:**
- [ ] 벡터와 행렬의 기본 연산
- [ ] 행렬 곱셈과 전치 (Transpose)
- [ ] 역행렬 (Inverse Matrix)
- [ ] 행렬식 (Determinant)
- [ ] 벡터 공간과 부분공간
- [ ] 선형 독립과 기저 (Basis)
- [ ] 고유값과 고유벡터 (Eigenvalues & Eigenvectors)
- [ ] 내적 (Dot Product)과 외적 (Cross Product)

**권장 선수과목:**
- 대학 1-2학년 선형대수학
- MIT 18.06 Linear Algebra 수준

**자가 진단 문제:**
```
1. 다음 행렬 A의 고유값을 구하시오:
   A = [[2, 1],
        [1, 2]]

2. rank(A) = 2, A ∈ ℝ^(3×3) 일 때, null(A)의 차원은?

3. 두 벡터 u, v가 직교할 필요충분조건은?
```

#### 1.2 미적분학 (Calculus)
**필수 개념:**
- [ ] 극한과 연속성
- [ ] 미분 (Differentiation)
- [ ] 편미분 (Partial Derivatives)
- [ ] 체인 룰 (Chain Rule)
- [ ] 적분 (Integration)
- [ ] 테일러 급수 (Taylor Series)
- [ ] 그래디언트 (Gradient)
- [ ] 헤시안 행렬 (Hessian Matrix)

**권장 선수과목:**
- 미적분학 I, II
- 다변수 미적분학

**자가 진단 문제:**
```
1. f(x,y) = x²y + y³에 대해 ∂f/∂x와 ∂f/∂y를 구하시오.

2. f(g(h(x)))의 도함수를 체인 룰로 표현하시오.

3. f(x) = e^x를 x=0 근처에서 3차 테일러 급수로 전개하시오.
```

#### 1.3 확률 및 통계 (Probability & Statistics)
**필수 개념:**
- [ ] 확률 공간과 확률 변수
- [ ] 확률 분포 (이산, 연속)
- [ ] 기댓값 (Expectation)과 분산 (Variance)
- [ ] 조건부 확률 (Conditional Probability)
- [ ] 베이즈 정리 (Bayes' Theorem)
- [ ] 주요 분포: 가우시안, 베르누이, 이항, 포아송
- [ ] 중심극한정리 (Central Limit Theorem)
- [ ] 최대우도추정 (Maximum Likelihood Estimation)

**권장 선수과목:**
- 확률론 입문
- 수리통계학

**자가 진단 문제:**
```
1. X ~ N(0,1), Y ~ N(0,1)이고 독립일 때, X+Y의 분포는?

2. 동전을 10번 던져 앞면이 7번 나올 확률은? (공정한 동전)

3. P(A|B) = 0.8, P(B) = 0.3, P(A) = 0.5일 때, P(B|A)는?
```

---

## 🟡 권장 사전 지식 (Recommended)

### 2. 프로그래밍

#### 2.1 Python 기초
**필수 스킬:**
- [ ] 기본 문법 (변수, 함수, 클래스)
- [ ] 리스트, 딕셔너리, 튜플 자료구조
- [ ] 반복문과 조건문
- [ ] 파일 입출력
- [ ] 예외 처리
- [ ] 기본적인 객체지향 프로그래밍

**자가 진단:**
```python
# 다음 코드를 이해할 수 있나요?
class Model:
    def __init__(self, params):
        self.params = params

    def forward(self, x):
        return sum(p * xi for p, xi in zip(self.params, x))

model = Model([1, 2, 3])
result = model.forward([4, 5, 6])  # result = ?
```

#### 2.2 NumPy
**필수 스킬:**
- [ ] 배열 생성 및 조작
- [ ] Broadcasting
- [ ] 배열 인덱싱과 슬라이싱
- [ ] 수학 연산 (sum, mean, std, etc.)
- [ ] 선형대수 함수 (dot, matmul, linalg 모듈)

**자가 진단:**
```python
import numpy as np

# 다음을 이해할 수 있나요?
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = A @ B  # C의 shape는?

# Broadcasting을 이해하나요?
x = np.array([[1, 2, 3]])  # shape: (1, 3)
y = np.array([[1], [2]])   # shape: (2, 1)
z = x + y  # z의 shape는?
```

#### 2.3 Matplotlib (선택)
- [ ] 기본 플롯 생성
- [ ] 여러 subplot 사용
- [ ] 레이블 및 범례 추가

---

## 🟢 유용한 배경 지식 (Nice-to-Have)

### 3. 기계학습 기초

#### 3.1 ML 기본 개념
- [ ] 지도학습 vs 비지도학습
- [ ] 훈련/검증/테스트 분할
- [ ] Overfitting과 Underfitting
- [ ] Cross-validation
- [ ] 기본적인 평가 지표 (Accuracy, Precision, Recall)

#### 3.2 전통적 ML 알고리즘 (선택)
- [ ] 선형 회귀 (Linear Regression)
- [ ] 로지스틱 회귀 (Logistic Regression)
- [ ] k-NN
- [ ] Decision Tree
- [ ] SVM 기초

### 4. 딥러닝 입문 (선택)

**알고 있으면 도움이 되는 개념:**
- [ ] 신경망 기본 구조
- [ ] 활성화 함수 (ReLU, Sigmoid, Tanh)
- [ ] 손실 함수 (MSE, Cross-Entropy)
- [ ] Gradient Descent 기본 개념
- [ ] Backpropagation의 직관적 이해

**이 강의에서 배울 내용이므로 몰라도 괜찮습니다!**

---

## 📊 자가 평가 가이드

### Level 0: 준비 부족 ⚠️
- 필수 사전 지식의 50% 미만 체크
- **권장 조치**: 선형대수학, 미적분학, 확률론 기초 강의 먼저 수강

### Level 1: 최소 준비 완료 ✅
- 필수 사전 지식의 50-70% 체크
- 권장 사전 지식 일부 보유
- **학습 방법**: 모르는 부분은 병행하여 학습하며 진행

### Level 2: 잘 준비됨 🎯
- 필수 사전 지식의 70-90% 체크
- 권장 사전 지식 대부분 보유
- **학습 방법**: 표준 진도로 학습 가능

### Level 3: 매우 잘 준비됨 🚀
- 필수 사전 지식 90% 이상 체크
- 권장 및 유용한 배경 지식 보유
- **학습 방법**: 심화 내용과 프로젝트에 집중

---

## 📚 추천 사전 학습 자료

### 선형대수학
1. **3Blue1Brown - Essence of Linear Algebra** (YouTube)
   - 시각적 직관 개발에 최고
   - 무료, 한글 자막 가능

2. **MIT 18.06 - Linear Algebra** (Gilbert Strang)
   - OCW에서 무료 수강
   - 명강의로 유명

3. **Linear Algebra Done Right** (Sheldon Axler)
   - 이론적 깊이가 필요하면

### 미적분학
1. **MIT 18.01 - Single Variable Calculus**
2. **MIT 18.02 - Multivariable Calculus**
3. **Khan Academy - Calculus** (한글 지원)

### 확률 및 통계
1. **MIT 6.041 - Introduction to Probability**
2. **Joe Blitzstein - Stat 110** (Harvard)
3. **확률과 통계** (이상화 외, 한국어)

### Python & NumPy
1. **Python for Everybody** (Coursera)
2. **NumPy Documentation Tutorial**
3. **Python Data Science Handbook** (Jake VanderPlas)

---

## 🎓 학습 경로 추천

### 경로 1: 수학 기초가 약한 경우
```
1. 선형대수학 복습 (2-4주)
   → 3Blue1Brown 시리즈 시청
   → MIT 18.06 핵심 강의

2. 미적분학 복습 (2-4주)
   → 다변수 미적분 집중
   → Chain rule, Gradient 개념 확실히

3. 확률론 기초 (2-3주)
   → 주요 분포 이해
   → 베이즈 정리 숙달

4. Python/NumPy 실습 (1-2주)
   → 코딩 연습

5. AI Mathematics Study 시작!
```

### 경로 2: 수학은 되는데 코딩이 약한 경우
```
1. Python 기초 (1주)
   → 문법 숙달

2. NumPy 집중 (1주)
   → 배열 연산 연습
   → 선형대수 함수들

3. Jupyter Notebook 익히기 (2-3일)

4. AI Mathematics Study 시작!
   → 코드 예제를 직접 타이핑하며 학습
```

### 경로 3: 기초가 탄탄한 경우
```
AI Mathematics Study 바로 시작!
→ 부족한 부분은 병행 학습
→ 고급 주제에 더 많은 시간 투자
```

---

## ❓ FAQ

### Q1: 미적분학을 오래 전에 배워서 기억이 안 나는데 괜찮을까요?
**A**: 괜찮습니다! 각 섹션마다 필요한 수학 개념을 복습할 수 있도록 했습니다. 다만, 체인 룰과 편미분은 확실히 알고 시작하는 것을 권장합니다.

### Q2: Python은 할 줄 아는데 NumPy는 처음입니다.
**A**: NumPy는 학습하면서 익혀도 충분합니다. 공식 튜토리얼을 하루 정도 보고 시작하세요.

### Q3: 딥러닝을 전혀 몰라도 되나요?
**A**: 네! 이 강의는 딥러닝의 수학적 기초부터 다룹니다. 오히려 너무 많이 알면 "왜?"라는 질문 없이 넘어갔던 부분들을 다시 배우게 됩니다.

### Q4: 수학을 전공했는데 코딩을 못 해요.
**A**: Python 기초만 익히면 됩니다. 수학적 이해가 뛰어나다면 코드는 금방 따라올 것입니다.

### Q5: 학부생인데 들을 수 있나요?
**A**: 학부 3-4학년이고 위의 필수 과목들을 수강했다면 가능합니다. 다만 난이도가 높으므로 여유 있는 학기에 도전하세요.

---

## 🚀 준비되셨나요?

위의 체크리스트를 확인하고:

- ✅ **70% 이상** 체크: [AI_Mathematics_Guide.md](./AI_Mathematics_Guide.md)부터 시작!
- ⚠️ **50-70%** 체크: 부족한 부분 보완하며 천천히 시작
- ❌ **50% 미만**: 추천 자료로 기초 다지기

**행운을 빕니다! 🎓**

---

**다음 단계**: [LEARNING_PATH.md](./docs/LEARNING_PATH.md) - 상세한 학습 로드맵 확인
