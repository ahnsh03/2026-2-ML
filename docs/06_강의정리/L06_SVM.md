# L06. 서포트 벡터 머신(SVM)

> 강의노트: `Lecture Note/Lecture 6 - SVM.pdf` (2025 작년 노트, 31쪽) · 올해 7주차
> 작년 강의영상: "Lecture 6 - Support vector machine" (`qhjV53AkW-g`) — 자동 자막 기반 · 기준일 2026-09-25

## 한눈에

- 이진 분류기를 세 단계로 쌓는다: **maximal margin classifier**(선형 분리 가능할 때만) → **support vector classifier**(soft margin, 경계는 선형) → **SVM**(커널로 비선형 경계).
- 초평면 `wᵀx + b = 0`, 점과 초평면의 거리 `|wᵀx + b| / ‖w‖`, margin과 support vector를 정의한다.
- margin 최대화를 **볼록 QP** `min ‖w‖² s.t. y_i(wᵀx_i + w0) ≥ 1`로 바꿔 유일한 해를 얻는다.
- slack `ε_i`와 위반 총예산 `C`로 bias–variance를 조절한다. **슬라이드의 C는 "예산"이라 scikit-learn의 C와 방향이 반대다.**
- 커널(다항·RBF·sigmoid)과 조합 규칙, RBF의 `σ`와 bias–variance, 다중 클래스(OvO/OvA)를 다룬다.
- RKHS·재생 성질·Representer 정리로 "보이지 않는 feature 공간에서의 최적화가 수학적으로 정당하다"를 보인다.

## 핵심 내용

### 1. 세 분류기의 관계 (p.2)

| 분류기 | 데이터 조건 | 결정 경계 |
|---|---|---|
| maximal margin classifier | 선형 분리 가능해야 한다 | 선형 |
| support vector classifier | 선형 분리 불가능해도 된다 | 선형 |
| SVM | 위 둘의 일반화 | 비선형(커널) |

### 2. 초평면 (p.3–6)

- 초평면은 주변 공간보다 한 차원 낮은 평평한 **affine 부분공간**이다(ℝⁿ에서 n−1차원). 부분공간은 덧셈·스칼라곱에 닫혀 있어야 하지만, affine 부분공간은 영벡터를 포함할 필요가 없어 닫혀 있지 않다.
- 정의: `wᵀx + b = 0`. 2D는 직선 `w_x x + w_y y + b = 0`, 3D는 평면 `w_x x + w_y y + w_z z + b = 0`.
- `wᵀx + b > 0` 또는 `< 0`으로 점이 어느 쪽에 있는지 정해진다 → 공간을 둘로 나눠 이진 분류에 쓴다. 핵심 가정: 클래스가 선형 결정 경계로 나뉜다.
- 로지스틱 회귀도 사실상 분리 초평면을 찾는다(p.5, 시그모이드의 0.5 임계값).
- 선형 분리 가능하면 분리 초평면은 무수히 많다. **train에서는 모두 같은 성능이지만 test(일반화)에서는 결과가 다르다.** 어떤 것이 이상적인가, 새 점은 어디에 속해야 하는가?

### 3. Maximal margin hyperplane (p.7–11)

- 거리 계산
  - bias 없는 `wᵀx = 0`: `w`는 초평면에 수직, 단위방향 `w/‖w‖`, 점 `x_i`의 거리 `|wᵀx_i| / ‖w‖`.
  - bias 있는 `wᵀx + b = 0`: 평행이동한 원점 초평면 `HP_o`를 두고 `d(HP, x_i) + d(HP, HP_o) = d(x_i, HP_o)`, `d(HP, HP_o) = b/‖w‖` → 거리 `|wᵀx_i + b| / ‖w‖`.
- **margin** = 모든 학습점 중 초평면까지의 최소 거리. **support vector** = 거리가 margin과 같은 점. **maximal margin hyperplane** = 학습점에서 가장 먼 분리 초평면.
- 최적화 (`(x_i, y_i)`, `y_i ∈ {−1, 1}`)
  - `max_w M  s.t.  Σ_j w_j² = 1,  y_i(wᵀx_i + w0) ≥ M  ∀i` — `‖w‖` 고정은 스케일 모호성을 없애 문제를 well-defined하게 만드는 제약이다.
  - ⇔ `min_w ‖w‖²  s.t.  y_i(wᵀx_i + w0) ≥ 1` (margin을 1로 고정하고 `‖w‖`를 푼 형태. 이때 margin = `1/‖w‖`)
- 이차 비용 + 선형 제약 = **이차계획(QP), 볼록** → 전역 최소가 유일하다. 일반형 `min_x ½xᵀP0x + q0ᵀx + c0  s.t.  Ax = b, x ≥ 0`.
- 한계(p.11): 분리 초평면이 없으면 쓸 수 없다. 개별 점에 민감하고, 학습 데이터에 과적합할 수 있으며, **이상치 하나가 알고리즘을 망칠 수 있다.**

### 4. Support vector classifier — soft margin (p.12–16)

- 학습점이 margin 안이나 초평면 반대편("wrong side")에 있어도 허용한다. maximal margin hyperplane은 **hard-margin SVM**, 이것은 **soft-margin SVM**이다.
- margin은 더 이상 최소 거리가 아니라 파라미터로 정해진다. support vector는 **margin 안 또는 반대편에 있는 점**이다.
- `max_w M  s.t.  Σ_j w_j² = 1,  y_i(wᵀx_i + b) ≥ M(1 − ε_i) ∀i,  Σ_i ε_i ≤ C,  ε_i ≥ 0 ∀i`
  - 모든 점은 초평면에서 M 이상 떨어져 있거나 벌점 `ε_i`를 낸다. 벌점 총합은 C로 제한된다.

| slack | 의미 |
|---|---|
| `ε_i = 0` | margin 바깥, 올바른 쪽 |
| `ε_i > 0` | margin을 위반 |
| `ε_i > 1` | 초평면 반대편 = 오분류 |

- **C = 위반 총예산**이며 교차검증으로 튜닝하는 하이퍼파라미터다. `C = 0`이면 maximal margin classifier가 된다. C를 키워 가면 bias–variance tradeoff가 나타난다.

| | 큰 C (예산 많음) | 작은 C (예산 적음) |
|---|---|---|
| bias / variance | high bias, low variance | low bias, high variance |
| margin | 넓다 | 좁다 |
| support vector | 많다 | 적다 |

- (보충) **scikit-learn의 `C`는 반대 방향이다.** `SVC`는 `min ½‖w‖² + C Σ_i ξ_i  s.t.  y_i(wᵀφ(x_i) + b) ≥ 1 − ξ_i`를 풀며, 여기서 C는 **위반 벌점의 가중치**다. sklearn에서 C를 키우면 margin이 좁아지고 support vector가 줄며 low bias / high variance가 된다(슬라이드의 예산 C와 대략 반비례).
- 여전히 선형 경계다(p.16): 1·3사분면과 2·4사분면으로 나뉜 데이터는 직선 하나로 못 나누지만, `x1x2`라는 축을 더하면 평면으로 나뉜다.

### 5. Feature 공간 확장 (p.17–18)

- `x = (x1, …, xn) ∈ ℝⁿ → x̃ = (x1, …, xn, x1², …, xn²) ∈ ℝ^{2n}`에서 support vector classifier를 풀면, 2n차원의 초평면이 원래 공간에서는 비선형 경계(이 예에서는 타원)가 된다.
- 고차 항을 계속 더하면 feature 수가 폭증해 계산이 어려워진다 → 많은 feature를 효율적으로 다룰 방법이 필요하다 = 커널.

### 6. SVM과 커널 (p.19–24)

- SVM은 support vector classifier에 커널을 써서 비선형 경계를 얻는다. 커널 `K(x_i, x_j) = φ(x_i)ᵀφ(x_j)`는 데이터를 고차원으로 **암묵적으로** 매핑한다.
- 선형 SVM의 해는 Lagrangian 최적성 조건에서 `w = Σ_i α_i y_i x_i` → 분류기 `f(x) = wᵀx + b − M = Σ_i α_i y_i x_iᵀx + b − M`. 데이터는 **내적으로만** 등장한다.
- 일반 SVM은 내적을 커널로 바꾼다: `f(x) = Σ_i α_i y_i K(x_i, x) + b − M`, `f(x) > 0`이냐 `< 0`이냐로 분류한다.
  - (보충) `α_i > 0`인 점이 support vector뿐이라 예측 비용은 support vector 수에 비례한다.
- 커널의 성질(p.21): 매핑 `φ: X → X^φ`를 계산하지 않는 내적의 일반화, **대칭** `K(x, x') = K(x', x)`, 커널 행렬은 **양의 준정부호(PSD)** `K ⪰ 0`.
- 대표 커널: 다항 `(x_iᵀx_j + c)^d`, RBF(Gaussian) `exp(−‖x_i − x_j‖² / 2σ²)`, sigmoid `tanh(α x_iᵀx_j + c)`, 그 밖에 Fisher, neural tangent, Laplacian, Bessel.
- 조합 규칙: `K1 + K2`, `αK1`, `K1·K2`도 커널이다.
- **RBF 상세**(p.23)
  - 1차원에서 `φ(x) = exp(−x²/2σ²)·[1, x/(σ√1!), x²/(σ²√2!), x³/(σ³√3!), …]ᵀ` — 무한 차원 벡터라 아무도 직접 계산하지 않는다.
  - 실무에서 매우 인기 있다: 매우 매끄러운 가설을 주고, k-최근접 이웃과 비슷하지만 더 매끄러우며, 다항식보다 덜 진동한다.
  - `σ`는 validation으로 고른다. **큰 σ → 넓은 Gaussian, 매끄러운 h → more bias, less variance.**
  - (보충) scikit-learn은 `gamma = 1/(2σ²)`를 쓴다. gamma가 클수록 σ가 작아져 high variance가 된다. `h`는 가설(결정 함수)이다.
- 커널별 결정 경계(p.24)

| 커널 | 특징 |
|---|---|
| linear | 선형 분리 데이터에 최적, 가장 단순하고 해석이 쉽다 |
| polynomial | 곡선 경계, 차수가 오를수록 유연해진다 |
| RBF | 매우 유연하고 국소적인 경계 |
| sigmoid | 신경망에서 착안했으나 불안정해 거의 쓰지 않는다 |

### 7. 3개 이상 클래스 (p.25)

| 방식 | 학습·평가할 SVM 수 | 특징·단점 |
|---|---|---|
| One vs. one | 클래스 쌍마다 1개, `k(k−1)/2` | k가 크면 계산이 비싸다. 예측은 다수결 |
| One vs. all | 클래스마다 나머지 전체와 1개, `k` | **클래스 불균형을 키울 수 있고, 초평면까지의 거리가 확신도와 잘 대응하지 않을 수 있다** |

### 8. 요약 (p.26)

- 장점: 정규화 파라미터 C가 과적합을 막는다, 커널로 경계 모양이 유연하다, **최적화가 볼록이라 해가 유일하다.**
- 단점: 하이퍼파라미터(C, 커널 함수)를 반드시 튜닝해야 한다, 이진 분류로 정식화해야 한다, 해석이 어렵다.

### 9. RKHS — 커널 방법의 수학적 정당성 (p.27–30)

- 문제의식: 커널 트릭에서는 feature map을 명시적으로 모른다. 모르는 공간에서 최적화·정규화·학습을 해도 수학적으로 괜찮은가?
- **Hilbert 공간**: Cauchy 수열의 극한을 포함하는(완비) 내적 공간. 유클리드 공간을 함수로 일반화한 것이다.
  - 내적 `⟨·,·⟩_H: H × H → ℝ`은 선형 `⟨α1f1 + α2f2, g⟩ = α1⟨f1, g⟩ + α2⟨f2, g⟩`, 대칭 `⟨f, g⟩ = ⟨g, f⟩`, `⟨f, f⟩ ≥ 0`이고 `f = 0`일 때만 0.
  - Cauchy 수열 예: 1, 1.4, 1.41, 1.414, 1.4142, … → √2. 극한 √2가 ℚ에 없으므로 ℚ는 완비가 아니고, ℝ은 완비다.
- **재생 성질**: `f(x) = ⟨f(·), k(·, x)⟩_H`. f가 숨겨져 있어도 커널과의 내적으로 `f(x)`를 얻는다. PSD 커널은 재생 커널을 정의한다(**Moore–Aronszajn 정리**).
- **Representer 정리**: RKHS 위의 정규화된 경험 위험 최소화 문제에서 최적 함수는 항상 학습점에 놓인 커널의 선형결합이다. `f̂(·) = Σ_i α_i k(x_i, ·)`, `f̂(x) = Σ_i α_i k(x_i, x)`.
- 정리(p.30): 대칭·PSD 커널이면 RKHS가 정의된다(함수 공간을 따로 정의할 필요가 없다). 학습 문제에서 무한 차원 함수를 뒤질 필요 없이 학습점 커널의 가중합만 찾으면 된다.
- **RKHS 노름** `‖f‖_H² = ⟨f(·), f(·)⟩_H`은 함수 복잡도의 내장 척도다. **노름이 작으면 매끄러운 함수 → 더 나은 일반화**(p.30 그림: 들쭉날쭉한 적합 vs 매끄러운 적합).
- (보충) 12주차 Gaussian Process도 같은 커널(RBF 등)을 쓰고, 예측 평균이 학습점 커널의 가중합이라는 같은 형태를 갖는다.

## 슬라이드의 코드·라이브러리

없음. 슬라이드에 코드는 없고 참고 자료만 있다(p.31): Stanford CME250 lecture 5, Princeton COS495 SVM I·II 슬라이드, Gatsby(UCL) RKHS 강의.

(보충) 슬라이드 개념에 대응하는 scikit-learn API는 다음과 같다.

| 슬라이드 개념 | scikit-learn (보충) | 비고 |
|---|---|---|
| support vector classifier(선형) | `LinearSVC`, `SVC(kernel='linear')` | `LinearSVC`는 표본이 많을 때 빠르다(기본 손실은 squared hinge) |
| 커널 SVM | `SVC(kernel=..., C, gamma, degree, coef0)`, kernel은 `'rbf'` · `'poly'` · `'sigmoid'` | `gamma='scale'`(기본) = `1/(n_features·X.var())` |
| 다중 클래스 | `SVC`는 내부적으로 OvO, `LinearSVC`는 OvR | `SVC`의 `decision_function_shape='ovr'`은 출력 모양만 바꾼다 |
| 확률 출력 | `SVC(probability=True)`(내부 5-fold Platt) 또는 `CalibratedClassifierCV(method='sigmoid')` | 결정값 자체는 확률이 아니다 |
| support vector | `support_vectors_`, `n_support_`, `dual_coef_`(= `α_i y_i`) | SV 수 진단 |

## 교수님이 강조한 것 (작년 영상)

1. **목표는 train이 아니라 test 일반화** — "얘네들은 모두 트레이닝에 대해서 완벽하게 … 나누기 때문에 정확하게 같은 성능 … 테스트셋에 대해서는 … 모두 다 다른 성능을 가지게 되는 거고 우리는 테스트셋에도 더 잘되는 형태의 클래시파이어를 얻고 싶어요." → margin 최대화의 동기가 일반화임을 분명히 한다.
2. **hard margin은 가장 가까운 한두 점에 좌우된다** — "가장 가까운 데이터 한 포인트에 대해서만 사용하기 때문에 … 플레인이 크게 크게 바뀔 수가 있습니다. 트레이닝 데이터에 조금 더 오버핏될 수가 있고" → 여러 점으로 경계를 정하는 soft margin이 더 robust하다는 논리로 넘어간다.
3. **C는 "얼마나 틀려도 되냐"의 예산** — "C가 결국 얼마만큼 많이 틀려도 되냐 … 많이 틀려도 되게 해 준다고 하면은 말진이 당연히 커지겠죠", "크게 변하지 않는다는 것은 베리언스가 적다는 것입니다. 대신 그만큼 바이어스가 커요." → 슬라이드 정의 기준의 설명이다. 코드(sklearn)는 C 방향이 반대이므로 보고서에서 어느 C인지 밝힌다.
4. **QP·볼록이라 효율적이고 해가 유일** — "컨벡스 옵티마이제이션을 굉장히 효율적으로 풀 수가 있어", 요약에서 "옵티마이제이션 프로블럼이 컨벡스한 형태여서 유니크한 솔루션을 만들어 줍니다." → 4주차 최적화와 연결된다. SVM 결과는 초기값에 따라 흔들리지 않는다.
5. **φ를 몰라도 커널만 알면 된다, 커널은 범용 도구** — "우리가 파이를 몰라도 … 커널만 알면 됩니다. 커널 펑션의 결과만 알면 돼요." "이 커널이라는 스킬은 다양한 머신러닝 알고리즘에서 사용이 될 수 있는 스킬이에요." 또 "저번 시간에 … 리니어 PCA만 했는데 … 커널 PCA까지 … 배우지를 못했는데"라며 커널을 이 시간에 처음 설명했다. → kernel PCA(6주차)와 GP(12주차)로 이어지는 공통 도구다.
6. **실무 기본값은 RBF, σ가 크면 bias↑ variance↓** — "일반적으로는 그냥 대부분 RBF를 사용합니다." "시그마가 크면 바이어스는 커지지만 베리언스가 적어진다라고 해석을 할 수가 있습니다." → 우리 SVM 단계는 linear vs RBF 비교로 충분하다. sklearn의 `gamma = 1/(2σ²)`라 방향이 뒤집힌다.
7. **OvA는 클래스 불균형에 약하다** — "데이터의 개수가 클래스마다 차이가 많이 날 수가 있고 … 인밸런스 때문에 하이퍼플레인이 뭐 잘 계산되지 않을 수도 있다." → 실패 유형 다중분류에서 `class_weight`가 필요하다.
8. **튜닝은 필수이고, 구현보다 옵션이 중요** — 요약에서 "버짓을 어떻게 설정할지 커널을 어떻게 설정할지에 대한 튜닝을 해 줘야 되고". L02-2에서도 "SK가 가지고 SVM 한 줄 치면은 구현이 되어 있거든요 … 우리가 알아야 될 것은 이 SVM으로 구현하는 이 한 줄의 코드에서 여러 가지 옵션들이 있을 거예요. 하이퍼파라미터 튜닝을 어떻게 해야 될지". → 직접 구현이 아니라 스케일링·C·커널·탐색 설계가 평가 포인트다(§7 단계 6과 연결).
9. **RKHS는 개념만** — "RKHS에서 … 계산이 되는 거지만 수학적으로 정당하다. 그래서 SVM이 정당하다라는 정도만 알면 될 것 같습니다." → 시험 대비 깊이의 힌트로 보인다(추정). 증명보다 재생 성질·Representer 정리의 의미를 설명할 수 있으면 된다.
10. **중간고사 직전 범위, 영상으로 다시 보라** — 다음 강의(L07) 첫머리: "다음 주 수요일 오전 9시에 머신 러닝 수업을 봅니다. 시험 여기서 봅니다." "(지난 시간) SVM … 이런 것들이 조금 까다롭긴 한데 한번 렉처 쭉 더 보면서 공부하면 될 거 같습니다." → 작년에도 SVM이 중간고사 직전 강의였다. 올해도 7주차 SVM 다음이 8주차 중간고사다.

## 우리 프로젝트(A1)에서 쓰는 곳

[구현 계획](../05_구현계획/A1_구현계획.md) §7 단계 2 "SVM (linear vs RBF, Platt 확률) — 비선형 경계"가 직접 대응하고, 단계 6 random vs grid 탐색의 대상 모델이기도 하다. 일정상 10주차 milestone의 베이스라인(0a·0b·LR·SVM)에 들어간다(§10).

| 강의 내용 | A1에서 쓰는 곳(계획 §) | 구현 메모(scikit-learn 클래스·하이퍼파라미터·주의점) |
|---|---|---|
| support vector classifier(선형 경계) | §7 단계 2, 단계 1 LR과 비교 | `LinearSVC(C=…, class_weight='balanced')` 또는 `SVC(kernel='linear')`. LR과 같은 fold·같은 feature로 PR-AUC를 비교한다. 둘이 비슷하면 "선형 한계"가 손실 함수가 아니라 경계 모양 때문임을 보여 준다 |
| 커널 SVM(RBF), 비선형 경계 | §7 단계 2 "비선형 경계", H3 | `SVC(kernel='rbf', C=…, gamma=…)`. linear 대비 향상폭이 "비선형성이 필요한가"의 답이다. feature 그룹(A+B+D, +C)별로 같이 돌려 ablation 표에 넣는다 |
| 커널은 거리 기반 → 스케일에 민감(L02 스케일링) | §5 feature, §8.1 분할 | `make_pipeline(StandardScaler(), SVC(...))`를 CV 안에서 fit한다(스케일러를 전체 데이터에 먼저 fit하면 누수). 한쪽으로 치우친 feature(예측 면적, Laplacian 분산 등)는 log 변환을 검토한다 |
| C(예산)·σ의 bias–variance, validation으로 선택 | §7 단계 6 random vs grid | 탐색은 로그 스케일: sklearn `C ∈ [1e-2, 1e3]`, `gamma ∈ [1e-4, 1e0]`(또는 `'scale'`의 배수). grid 7×7 = 49회와 `scipy.stats.loguniform` random 49회로 같은 예산을 비교한다. 슬라이드 C와 sklearn C가 반대임을 주석·보고서에 적는다 |
| "초평면까지의 거리가 확신도와 잘 대응하지 않는다"(p.25) | §7 공통 calibration, §8.2 ECE·Brier | `decision_function`은 확률이 아니다. PR-AUC·ROC-AUC·risk–coverage는 결정값으로 충분하다. ECE·reliability diagram에는 `CalibratedClassifierCV(estimator=pipe, method='sigmoid', cv=<run 단위 split 목록>)`로 Platt 보정한 확률을 쓴다. `SVC(probability=True)`는 내부 5-fold가 무작위라 같은 run이 섞이므로 피한다 |
| 클래스 불균형(OvA 경고) | §4 `y_fail`(val 10.4%, front 19.0%) | `class_weight='balanced'`. 지표는 accuracy가 아니라 PR-AUC(주)와 재현율 |
| 다중 클래스 OvO/OvA | §6 "쓸모": 실패 유형 다중분류 | GMM 군집 id를 타깃으로 `SVC`(내부 OvO) vs `LinearSVC`(OvR), `class_weight='balanced'`, macro-F1로 비교한다 |
| 해석이 어렵다 | §7 단계 3 importance, H3 | linear는 표준화된 feature의 `coef_`로 방향을 보고 LR 계수와 대조한다. RBF는 GBM과 같은 방식의 `permutation_importance`로 본다 |
| QP 계산량, support vector 수 | §3.3 데이터 규모, §8.2 비용 | RBF `SVC`의 학습 시간은 표본 수에 대해 적어도 제곱으로 늘어난다. F 묶음 3뷰 약 12,000 sample × GroupKFold × 탐색 49회는 무거우므로 탐색은 층화 서브샘플로, 최종 적합만 전체로 한다. `n_support_` 비율(높으면 클래스가 많이 겹침)과 학습 시간을 기록한다 |
| 커널·RKHS | §7 단계 5 GP, H5 | 같은 RBF 커널을 GP에도 쓴다. SVM은 점수만, GP는 예측분산까지 주므로 OOD(P5)에서의 불확실성 비교는 GP가 맡는다. SVM 결정값의 크기가 OOD에서 줄어드는지는 보조로만 본다. L11에서 교수님도 "SVM 같은 경우는 일직선으로 분류 … 커널을 넣기 시작하면 논리니어한 바운더리"를 모델 선택(epistemic 불확실성)의 예로 들었다 |

## 구현 체크리스트

- [ ] SVM은 항상 `Pipeline(StandardScaler, SVC)`로 만들고, GroupKFold(run 단위) 안에서만 fit했다
- [ ] 코드 주석에 "sklearn C = 벌점 가중치(슬라이드의 예산 C와 반대)", "gamma = 1/(2σ²)"를 적었다
- [ ] `class_weight='balanced'`를 켜고 PR-AUC(주)·ROC-AUC를 fold 평균 ± 표준편차로 보고했다
- [ ] linear와 RBF를 같은 fold·같은 feature 그룹으로 돌려 계단 표(§7)의 LR 옆에 넣었다
- [ ] 확률이 필요한 지표(ECE, Brier, reliability)는 run 단위 split으로 Platt 보정한 확률로 계산했고, 보정 전후를 함께 보였다
- [ ] grid와 random 탐색의 시도 횟수를 같게 맞추고 로그 스케일 탐색 공간을 기록했다(§7 단계 6)
- [ ] 학습 시간과 `n_support_`를 기록했고, 서브샘플을 썼다면 크기와 층화 기준을 적었다
- [ ] 도메인 이동 분할(P2~P5)의 테스트 쪽과 L 묶음을 튜닝에 쓰지 않았다

## 올해 강의와 다를 수 있는 점

- 이 노트는 **2025 작년 노트**다. 올해는 7주차(구현 계획 §10 일정상 10/14–10/20)에 다룬다. **올해 노트가 올라오면 수식 표기(특히 C의 정의)와 RKHS 분량을 대조해 갱신한다.**
- 작년에는 kernel PCA를 건너뛰어 커널 트릭이 SVM 시간에 처음 나왔다. 올해 6주차에 kernel PCA를 다루면 이 강의의 커널 설명은 짧아지거나 복습 형태가 될 수 있다.
- 작년처럼 SVM 다음 주가 중간고사(8주차, 10/21)다. 교수님이 SVM을 "까다롭다"고 했으므로 시험 준비는 C·σ의 bias–variance 방향, slack `ε_i` 해석, 커널 성질(대칭·PSD·조합 규칙), OvO/OvA 비교를 중심으로 한다.
- RKHS는 작년에 "개념만"이라고 했다. 올해 범위가 달라지면 이 절을 갱신한다.
