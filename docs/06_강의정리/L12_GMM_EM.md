# L12. 가우시안 혼합 모델(GMM)과 EM 알고리즘

> 강의노트: `Lecture Note/Lecture 12 - EM and GMM.pdf` (2025 작년 노트, 32쪽) · 올해 13주차
> 작년 강의영상: "Lecture 12 - Expectation maximization and GMM" (`h18VfzrZYHc`) — 자동 자막 기반 · 기준일 2026-09-25

## 한눈에

- hard clustering(K-means, DBSCAN, 계층 군집)은 각 점을 한 군집에만 넣는다. 이 강의는 각 점이 군집마다 속할 **확률**을 주는 soft clustering을 다룬다.
- GMM은 `p(x|θ) = Σⱼ wⱼ N(x|μⱼ, Σⱼ)`이다. 잠재변수 z(군집 ID)를 도입하면 z의 사후확률, 곧 **responsibility γ**가 soft 할당이 된다.
- GMM의 MLE는 singularity, 식별 불가능성(K!개의 같은 해), non-convex 때문에 직접 풀기 어렵다. 그래서 **EM**을 쓴다. E-step에서 γ를 계산하고 M-step에서 파라미터를 갱신하기를 반복한다.
- EM은 **K-means의 soft 버전**이다. 둘 다 local optimum에 빠질 수 있고, 초기값에 민감하고, K를 미리 정해야 한다. K는 MDL·AIC·BIC 같은 기준으로 고른다.
- 일반 잠재변수 모델로 넓히면 Jensen 부등식에서 **ELBO**가 나오고, `log p(X|θ) = ELBO + KL(q ‖ p(z|X,θ))`가 성립한다. E-step은 KL을 0으로 만들어 bound를 딱 맞추고, M-step은 ELBO를 올린다. 그 결과 likelihood가 단조 증가한다.
- 사후분포를 계산할 수 없는 복잡한 모델은 variational inference(→ VAE)로 넘어간다.

## 핵심 내용

### 1. Hard vs soft clustering (p.3)

- hard clustering: 각 객체는 군집 하나에만 속한다(K-means, DBSCAN, hierarchical).
- soft clustering: 객체가 각 군집에 속할 확률을 준다.
- 교수님 설명: 경계에 걸친 애매한 점에 대해 "이 정도 불확실성을 가지고 있다"고 알려 주면 뒤에서 쓸모가 있고, 이는 지난주 UQ와 이어진다.

### 2. 가우시안 분포와 혼합 모델 (p.4–7)

- 중심극한정리: 작은 효과가 많이 더해진 결과는 가우시안에 가까워진다. 교수님은 Galton board로 설명했다.
- 1차원 가우시안은 `p(x) = 1/(σ√(2π)) · exp(−(x−μ)² / (2σ²))`, d차원 가우시안은 `p(x) = 1/√((2π)^d |Σ|) · exp(−½ (x−μ)ᵀ Σ⁻¹ (x−μ))`이다.
- iris 데이터의 petal/sepal length는 품종별로 보면 가우시안이고, 섞어 보면 봉우리가 여럿이다. 영상에서는 남녀가 섞인 사람 키, 사람·개·고양이가 섞인 집 안 생물의 키도 예로 들었다.
- 혼합 모델은 여러 pdf의 가중합이다: `p(x) = π₀f₀(x) + π₁f₁(x) + … + πₖfₖ(x)`, `Σπᵢ = 1`. 가중치 합이 1이어야 전체가 pdf가 된다.

### 3. GMM과 MLE (p.8–11)

- 데이터 `X = {x₁, …, xₙ}`에서 `θ = {w, μ, Σ}`(군집 확률, 군집 평균, 군집 공분산)를 찾는다. MLE는 `θ* = argmax_θ p(X|θ) = argmax_θ Πᵢ p(xᵢ|θ)`이다. 베이즈 복습으로 `p(belief|data) ∝ p(data|belief) p(belief)`도 다시 짚는다.
- GMM은 `p(xᵢ|θ) = Σⱼ₌₁ᴷ wⱼ N(xᵢ|μⱼ, Σⱼ)`, `Σwⱼ = 1`이다. 가우시안을 충분히 많이 쓰면 어떤 밀도든 근사할 수 있다(universal approximator).
- 잠재변수 zᵢ는 xᵢ가 나온 가우시안의 ID다.
  - `p(zᵢ = j) = wⱼ` (prior)
  - `p(xᵢ|zᵢ = j) = N(μⱼ, Σⱼ)`
  - `p(xᵢ|θ) = Σⱼ p(zᵢ = j) p(xᵢ|zᵢ = j)`
- log-likelihood는 `l(θ) = Σᵢ log Σⱼ wⱼ N(xᵢ|μⱼ, Σⱼ)`이다. log는 단조증가라 argmax가 바뀌지 않고, 곱을 합으로 바꿔 준다.
- 직접 최적화가 어려운 이유 (p.10)

| 문제 | 내용 |
|---|---|
| singularity | 가우시안 하나가 점 하나만 설명하며 σ → 0이 되면 likelihood가 무한대로 간다(spike 가우시안) |
| identifiability | 군집 번호를 바꿔도 같은 해다. 동등한 해가 K!개 있다 |
| non-convex | 전역 최적을 보장할 수 없다 |
| 제약조건 | `Σwⱼ = 1`, Σⱼ는 positive definite |

- z를 안다면 문제가 쉬워진다 (p.11): `l(θ) = Σᵢ [log N(xᵢ|μ_zᵢ, Σ_zᵢ) + log w_zᵢ]`. 군집마다 단일 가우시안을 적합하면 된다. 이때 `wⱼ → nⱼ/n`이고, μⱼ·Σⱼ는 그 군집에 속한 점들의 평균·공분산이다.

### 4. Responsibility (p.12)

- θ가 주어지면 zᵢ의 사후확률은 `γⱼ,ᵢ = p(zᵢ = j | xᵢ; θ) = wⱼ N(xᵢ|μⱼ, Σⱼ) / Σₖ wₖ N(xᵢ|μₖ, Σₖ)`이다.
- 이를 responsibility라 부른다. 군집 j가 점 i에 대해 지는 책임, 곧 점 i가 군집 j에 속할 확률이다.

### 5. EM 알고리즘 (p.13–17)

- **E-step**: 현재 모델에서 z의 사후확률 γ를 계산한다. 각 가우시안이 각 점을 얼마나 만들어 냈다고 볼지를 정하는 단계다.
- **M-step**: 데이터가 정말 그렇게 만들어졌다고 보고, 각 가우시안의 파라미터를 MLE로 갱신한다.
- (보충) GMM의 M-step 식은 p.11의 hard 버전을 γ로 가중한 것이다.
  - `Nⱼ = Σᵢ γⱼ,ᵢ`, `wⱼ = Nⱼ / n`
  - `μⱼ = Σᵢ γⱼ,ᵢ xᵢ / Nⱼ`
  - `Σⱼ = Σᵢ γⱼ,ᵢ (xᵢ−μⱼ)(xᵢ−μⱼ)ᵀ / Nⱼ`
- K-means와 비교 (p.14)

| | K-means | EM (GMM) |
|---|---|---|
| 할당 | 가장 가까운 중심 하나 (가중치 0 또는 1) | 사후확률 γ (soft) |
| 갱신 | 할당된 점들의 무게중심 | γ로 가중한 평균 + 공분산 + 군집 가중치 |
| 군집 모양 | 직선 경계, 원형 군집 | 공분산에 따른 타원 |
| 관계 | — | "prior와 공분산을 고정한 K-means의 soft 버전" |

- p.15 그림은 초기 원 두 개에서 출발해 L = 1, 2, 5, 20번 반복하며 수렴하는 과정이다. p.16 그림에서 길게 늘어진 군집을 K-means는 잘못 자르고 GMM은 제대로 나눈다.
- 문제점 (p.17)
  - local optimum으로 수렴한다.
  - 초기조건에 매우 민감하다.
  - 가우시안 개수를 미리 정해야 한다. 최적 개수는 정보이론 기준인 MDL, AIC, BIC, MML 등으로 고른다.
- (보충) scikit-learn의 정의는 BIC = `−2 log L + p·log n`, AIC = `−2 log L + 2p`이다(p는 자유 파라미터 수). 둘 다 작을수록 좋고, BIC가 파라미터에 더 강한 벌점을 준다.

### 6. 일반 잠재변수 모델과 ELBO (p.18–24)

- 확률변수는 두 종류다. z는 관측되지 않는 hidden 변수, x는 관측 변수다. 결합분포는 `p(X, z|θ)`, marginal likelihood는 `p(X) = Σ_z p(X, z)`이다. GMM은 잠재변수 모델의 한 예다.
- 문제도 두 종류다. learning은 `θ* = argmax p(X|θ)`를 찾는 것이고, inference는 `p(zᵢ|xᵢ, θ)`를 구하는 것이다.
- 가우시안을 가정하지 않고 일반적으로 쓰면 `l(θ) = log Σ_z q(z) · p(X,z|θ)/q(z)`이다. 여기서 q는 임의의 PMF(`Σq(z) = 1`)다.
- convexity 복습 (p.20–21)
  - convex function은 `f(αx + (1−α)y) ≤ αf(x) + (1−α)f(y)`를 만족하고, `f'' ≥ 0`이면 convex다. log는 concave다.
  - Jensen 부등식: convex 함수면 `E[f(x)] ≥ f(E[x])`이고, concave 함수(log)면 부등호가 반대로 `f(E[x]) ≥ E[f(x)]`이다.
- 따라서 `log p(X|θ) ≥ Σ_z q(z) log(p(X,z|θ)/q(z)) = L(q, θ)`이다.
  - 좌변 log p(X|θ)를 evidence라 하고, 우변은 그 하한이라 **ELBO**(evidence lower bound)라 부른다.
  - EM과 variational method는 q와 θ에 대해 ELBO를 최대화한다.
- 알고리즘 (p.23): θ_old에서 시작해 `q* = argmax_q L(q, θ_old)`, 이어서 `θ_new = argmax_θ L(q*, θ)`를 반복한다. 그림에서는 빨간 log p 곡선 아래에 있는 파란·초록 하한 곡선을 차례로 끌어올린다.
- ELBO 분해 (p.24): `L(q, θ) = −KL[q(z) ‖ p(z|X,θ)] + log p(X|θ)`. 즉 `log p(X|θ) = L(q, θ) + KL[q(z) ‖ p(z|X,θ)]`이다.

### 7. 정보이론 기초 (p.25–26)

- 엔트로피 `H(p) = −Σ p(x) log p(x)`는 확률변수의 불확실성, 곧 그 불확실성을 없애는 데 필요한 정보량(coding cost)이다. 균등분포일 때 가장 크다.
- 교차 엔트로피 `H(p, q) = −Σ p(x) log q(x)`는 q로 참 분포 p를 근사할 때 드는 비용이다. 영상에서는 제비가 낮게 나는지(q)로 날씨(p)를 얼마나 설명할 수 있는가에 비유했다.
- KL divergence `KL[p ‖ q] = Σ p log(p/q) = −H(p) + H(p, q)`는 p 대신 q를 쓸 때 치르는 벌점이다.
  - Jensen 부등식으로 KL ≥ 0임을 보일 수 있다.
  - 비대칭이다: `KL[p‖q] ≠ KL[q‖p]`, `H(p,q) ≠ H(q,p)`.

### 8. EM이 왜 동작하는가 (p.27–30)

- **E-step** (θ 고정): KL이 0이 되도록 `q(z) = p(z|X, θ_old)`로 둔다. 그러면 하한이 θ_old에서 log-likelihood와 딱 맞는다(tight).
- **M-step** (q 고정): θ에 대해 ELBO를 최대화한다. KL ≥ 0이므로 log-likelihood는 적어도 ELBO가 오른 만큼 오른다. 따라서 likelihood가 **단조 증가**한다.
- General EM (p.29)
  - E-step: `q*(z) = p(z|X, θ_old)`, `J(θ) = L(q*, θ)`
  - M-step: `θ_new = argmax_θ J(θ)`
  - ELBO의 전역 최대는 log-likelihood의 전역 최대이기도 하다. 슬라이드에 적힌 `log p(z|X,θ*)`는 `log p(X|θ*)`를 잘못 쓴 것으로 보인다.
- Generalized EM (p.30): 각 단계를 끝까지 풀기 어려울 때 쓴다. E-step에서는 KL을 줄이는 q를, M-step에서는 `J(θ_new) > J(θ_old)`인 θ 아무것이나 찾는다. 이렇게만 해도 likelihood는 단조 증가한다.

### 9. 정리 (p.31)

- GMM
  - 군집마다 서로 다른 가우시안을 두는 확률적 군집화다.
  - 가우시안 대신 다른 분포를 써도 된다.
  - EM으로 최적화한다.
- EM
  - 여러 잠재변수 모델에 두루 쓰는 최적화 방법이다.
  - 하한을 계산하고 그 하한을 최적화하기를 반복한다.
  - 수렴하지만 local optimum일 수 있다.
  - `p(z|X,θ)`를 계산해야 하는데, 복잡한 모델에서는 불가능하다. 해법은 variational inference다.
- 개요(p.2)에 있는 확장 주제(HMM, factor analysis, mixture of experts)는 본문 슬라이드에 없다.

## 슬라이드의 코드·라이브러리

슬라이드에 코드는 없다. p.16 Performance 그림의 제목이 `GaussianMixture`, `KMeans`(scikit-learn 클래스 이름)이다. 대응하는 API를 (보충)으로 정리한다.

| API (보충) | 용도 |
|---|---|
| `sklearn.mixture.GaussianMixture(n_components, covariance_type, n_init, init_params, reg_covar, max_iter, tol, random_state)` | EM으로 GMM 적합 |
| `.predict_proba(X)` / `.predict(X)` | responsibility γ / argmax γ(hard 할당) |
| `.bic(X)`, `.aic(X)` | K와 공분산 형태 선택 |
| `.score_samples(X)` | 점별 `log p(x)` |
| `.converged_`, `.n_iter_`, `.lower_bound_` | EM 수렴 확인 |
| `sklearn.cluster.KMeans(n_clusters, init='k-means++', n_init)` | 기준선 |
| `sklearn.mixture.BayesianGaussianMixture` | variational 추론판. 필요 없는 성분의 가중치를 0 쪽으로 보낸다 |

`covariance_type`은 슬라이드에 없는 (보충) 내용이다. d는 입력 차원, K는 성분 수다.

| 값 | 의미 | 공분산 파라미터 수 |
|---|---|---|
| `full` | 성분마다 완전한 공분산 | K·d(d+1)/2 |
| `tied` | 모든 성분이 공분산 하나를 공유 | d(d+1)/2 |
| `diag` | 성분마다 대각 공분산 | K·d |
| `spherical` | 성분마다 σ²I 하나 | K (K-means에 가장 가깝다) |

## 교수님이 강조한 것 (작년 영상)

1. **soft clustering의 가치**: "확실하게 초록색인 애들은 초록색이라고 … 애매하게 초록색과 빨간색 중간 지점인 애들은 얘네들이 중간 지점이라는 걸 알려 준다면 그 정보가 우리에게 뭔가 나중에 도움이 될 수가 있습니다", "저번 시간에 배웠었던 언서턴티 퀀티피케이션이나 이런 것과도 결국 연관"
   → 군집 결과도 불확실성과 함께 보고하라는 뜻이다. L11과 L12를 잇는 고리다.
2. **가우시안 가정의 근거**: 중심극한정리. "사람의 키 … 맨과 우먼이 있으니까 완벽한 가우시안이 아니라"
   → 요소 하나하나는 가우시안이어도 섞이면 혼합분포가 된다는 것이 GMM의 동기다.
3. **universal approximator**: "가우시안이 충분히 있다고 가정을 하면은 표현할 수 있기 때문에"
   → 성분 수 K를 늘리면 무엇이든 맞출 수 있다. 그래서 K를 BIC 같은 벌점 기준으로 제한해야 한다(해석).
4. **singularity**: "시그마가 0으로 갔었을 때 … 얘의 확률이 무한으로 가요 … 말도 안 되는 데이터에 대한 설명이지만"
   → 실제 구현에서는 공분산 정규화(`reg_covar`)가 필요하다(해석).
5. **identifiability**: "같은 데이터를 설명할 수 있는 방법이 K 팩토리얼만큼 있다"
   → 군집 번호 자체에는 의미가 없다. 군집을 비교할 때는 순열에 불변인 지표를 써야 한다(해석).
6. **K-means와 비교**: "K-means는 선을 기반으로 최적화를 해야 되니까 … GMM의 경우에는 그런 거랑 상관없어", "K-means의 결국 확장형이라고 볼 수가 있어요"
   → 길게 늘어진 군집을 다룰 때 GMM이 유리하다. K-means를 기준선으로 두는 근거가 된다.
7. **초기값과 K 선택**: "굉장히 이니셜 컨디션에 영향을 많이 받아요. K-means도 … 보완한 게 [K-means++]", "저번에는 실루엣 메소드나 … 여기서는 MDL, AIC, BIC 뭐 이런 크리테리아를 이용해서 결정"
   → 자막은 "테이니스 클러스턴스"로 깨져 있는데 K-means++로 추정한다. 여러 번 초기화하고 BIC로 K를 고르라는 것이 강의가 권하는 절차다.
8. **일반화**: "레이턴트 베리어블을 이용해서 최적화를 하는 방식은 다음과 같이 제너럴하게 … 계산이 될 수가 있다", GMM은 그 특수한 경우이고, variational inference는 "VAE 베리에이셔널 오토인코더의 기초적인 근간"
   → EM을 GMM 전용이 아니라 잠재변수 모델 일반의 도구로 이해하라는 뜻이다.
9. **복습 권고**: "수학적인 내용도 조금 복잡하고 … 시간을 들이면서 식들도 보고 이해를 해야 될 거 같아요"
   → ELBO·KL 유도는 스스로 다시 따라가 봐야 한다.

- 이 영상에는 숙제·프로젝트·시험에 관한 발언이 없다.

## 우리 프로젝트(A1)에서 쓰는 곳

| 강의 내용 | A1에서 쓰는 곳(계획 §) | 구현 메모 |
|---|---|---|
| GMM soft clustering (p.9, p.31) | §6 단계 4 "PCA 점수에 GMM" | 고유 오류맵의 PCA 점수(1,920차원 → 상위 m개 성분)에 `GaussianMixture`를 적합한다. PCA를 반드시 먼저 한다. 1,920차원 full 공분산은 성분 하나에 파라미터가 약 184만 개라 적합할 수 없고 singular해진다 |
| K 선택: AIC·BIC (p.17) | §6 단계 4 "BIC로 2~12 탐색", §8.2 군집 지표 | K = 2..12 × `covariance_type` {full, diag} 격자에서 `bic(X)` 곡선을 그린다. 최솟값만 보지 말고 곡선의 팔꿈치, bootstrap ARI 안정성, 사람 분류와의 일치도를 함께 본다. AIC도 같이 그려 두 기준이 다르면 그 차이를 설명한다 |
| 공분산 형태 (보충) | §6 단계 4 "full vs diag" | PCA 점수는 전체로는 무상관이지만 군집 안에서는 상관될 수 있어 full이 맞을 수 있다. 표본이 적은 군집은 diag가 안정적이다. 파라미터 수가 그대로 BIC 벌점에 들어간다 |
| singularity (p.10) | §6 단계 1–2 오류맵 (희소, 비슷한 sample이 많음) | `reg_covar`(기본 1e-6)가 PCA 점수 스케일에 맞는지 확인한다(예: 1e-4~1e-3 비교). 정차 중복 프레임이 curation(§3.4)에서 빠졌는지도 확인한다. 같은 점이 여러 개 있으면 spike 성분이 생긴다 |
| 초기값 민감·local optimum (p.17) | §6 단계 4 | `n_init=10` 이상, `init_params='k-means++'`(scikit-learn 1.1 이상) 또는 `'kmeans'`, `random_state`를 고정한다. `converged_`를 확인한다. 여러 초기화 중 `lower_bound_`가 가장 높은 해가 자동으로 선택된다 |
| identifiability K! (p.10) | §6 검증 "bootstrap ARI", "사람 분류와 NMI·ARI" | 군집 번호는 실행할 때마다 바뀐다. 번호를 맞추지 말고 ARI·NMI(순열 불변)로 비교한다. 군집 이름은 평균 오류맵을 보고 사람이 붙인다 |
| responsibility γ (p.12) | §6 단계 5 갤러리, 검증 "쓸모" | `predict_proba`로 대표 sample(γ > 0.9)만 골라 갤러리에 싣고, 애매한 sample(max γ < 0.6)은 따로 표시한다. 군집 id를 feature로 넣는 실험에서는 hard id와 γ 벡터를 함께 비교한다(보충) |
| K-means 대비 (p.14, p.16) | §6 단계 4 "K-means를 기준선" | 같은 PCA 점수와 같은 K로 `KMeans(n_init=10)`와 GMM을 비교한다(실루엣, 사람 분류 ARI, 안정성). "spherical·같은 분산·hard 할당 GMM ≈ K-means"라는 관계를 발표에서 짚으면 강의와 바로 이어진다 |
| soft clustering = 불확실성 (영상, L11과 연결) | §6 갤러리, 발표 | 할당 엔트로피 `−Σ γ log γ`를 "실패 유형 불확실성"으로 쓴다. IoU가 낮은데 어느 유형에도 확신이 없는 sample은 "기타" 또는 새 유형의 후보로 본다 |
| ELBO·일반 EM (p.18–30) | 직접 구현하지 않음 | scikit-learn이 EM을 수행한다. 발표에서 "EM은 log-likelihood를 단조 증가시킨다"는 점을 `lower_bound_` 값으로 보이는 정도면 충분하다 |

## 구현 체크리스트

- [ ] 군집 대상(전체 sample인지 저IoU sample만인지, 뷰를 나눌지)을 먼저 정해 문서에 적는다.
- [ ] PCA 성분 수 m을 설명분산 기준으로 정하고, whiten 여부를 GMM과 K-means에 똑같이 적용한다. K-means와 diag GMM은 입력 스케일의 영향을 받는다.
- [ ] K = 2..12 × {full, diag}로 BIC(와 AIC) 곡선을 그리고, 선택 근거를 표로 남긴다.
- [ ] `n_init ≥ 10`, `random_state` 고정, 모든 적합에서 `converged_ == True`인지 확인한다.
- [ ] `reg_covar` 값을 기록하고, 가중치 wⱼ가 아주 작거나 분산이 거의 0인 성분(singularity 징후)이 없는지 본다.
- [ ] 군집 비교와 안정성 평가는 ARI·NMI로 한다(번호 매칭 금지).
- [ ] 갤러리에 responsibility를 표시하고, 애매한 sample의 비율을 보고한다.
- [ ] K-means 기준선을 같은 입력·같은 K로 돌려 비교표에 넣는다.

## 올해 강의와 다를 수 있는 점

- 이 노트는 2025년판(Lecture 12, 작년 11/12 업로드)이다. 올해는 13주차(11/25)에 배정되어 있다.
- 계획 §10에는 같은 주에 "GMM 실패 유형 확정"이 잡혀 있다. 강의를 듣고 나서 시작하면 늦으므로, 9주차 Clustering 이후에 이 노트를 기준으로 GMM을 미리 돌려 둔다.
- 개요(p.2)의 확장 주제(HMM, factor analysis, mixture of experts)는 작년 노트 본문에 없었다. 영상 첫머리에서도 "익스텐션까지는 갈지는 모르겠습니다"라고 했고 실제로 다루지 않았다. 올해 추가되면 보완한다.
- 작년 영상에는 숙제 공지가 없었다. 올해 과제가 나오는지 확인한다.
- 올해 강의노트가 올라오면 이 문서를 갱신한다.
