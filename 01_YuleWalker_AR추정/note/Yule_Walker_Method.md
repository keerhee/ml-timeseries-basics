---
title: AR 모형의 계수 추정 — 네 가지 방법과 Yule-Walker
audience: 비전공자·학부생
source: "ritvikmath, The Yule-Walker Method (YouTube, https://youtu.be/1230arWETzY)"
tags: [time-series, AR, yule-walker, burg, mle, ols, toeplitz, levinson-durbin]
---

# AR 모형의 계수 추정 — 네 가지 방법과 Yule-Walker

> [!summary] 결론 먼저
> - **무엇을 하나:** "오늘 값 = 어제·그제 값의 가중합 + 잡음"이라는 AR 모형에서 그 **가중치(계수)** 를 데이터로 알아낸다.
> - **방법은 네 가지:** ① OLS(회귀) ② Yule-Walker(자기상관 맞추기) ③ Burg(앞뒤 예측) ④ 최대우도 MLE(확률 최대화).
> - **Yule-Walker:** 데이터에서 "몇 칸 떨어진 값끼리 얼마나 닮았나(자기상관)"만 계산해 작은 연립방정식을 푼다. 가장 단순하고, 추정 모형이 늘 정상이다.
> - **차이는 경계에서:** 데이터가 길고 정상이면 네 방법의 답은 거의 같다. 짧거나 계수가 1에 가까우면 Yule-Walker가 가장 크게 빗나가므로 Burg·MLE를 쓴다.
> - **전제:** 시계열이 **정상(stationary)** 이어야 한다.

원자료: ritvikmath, *The Yule-Walker Method*. 영상은 Yule-Walker 하나를 다루지만, 이 노트는 먼저 방법 네 가지를 펼쳐 전체 그림을 보여 준 뒤 Yule-Walker를 자세히 설명한다. 영상 판서의 오류는 §3.2와 부록에서 바로잡았다. 계산 속도(Big-O) 이야기는 본문에서 빼 부록으로 옮겼다.

---

## 1. 기본 개념

### 1.1 먼저 알아둘 용어

| 용어 | 쉬운 뜻 | 기호 |
|---|---|---|
| 시계열 (time series) | 시간 순서로 기록한 값. 매일 주가, 매달 물가 | $x_1,\ldots,x_n$ |
| 시차 (lag) | 몇 칸 떨어진 과거인가. 어제는 시차 1, 그제는 시차 2 | $k$ |
| 백색잡음 (white noise) | 평균 0, 서로 아무 관련 없는 순수한 무작위 충격 | $\varepsilon_t$ |
| 자기공분산 | 한 시계열과 **자기 자신의 과거**가 같이 움직이는 정도 | $\gamma_k$ |
| 자기상관 (ACF) | 자기공분산을 −1~1로 표준화한 값 | $\rho_k=\gamma_k/\gamma_0$ |
| 정상성 (stationarity) | 시간이 흘러도 평균·분산·자기상관이 변하지 않는 성질 | |
| 추정 (estimation) | 데이터로 모르는 값(계수)을 알아내는 일. 추정값에는 ^(햇)을 붙인다 | $\hat\phi$ |

### 1.2 자기상관 — "어제가 오늘을 얼마나 닮았나"

![[figs/fig_intuition.png]]

위쪽 백색잡음은 어제와 오늘 사이에 관계가 없다(산점도가 동그랗게 퍼지고 $\hat\rho_1\approx0$). 아래쪽 AR(1)은 어제가 높으면 오늘도 높은 경향이 있다(산점도가 기울고 $\hat\rho_1\approx0.8$). 이 "닮음 정도"가 자기상관이다.

### 1.3 AR($p$) 모형

$$
X_t=\phi_1X_{t-1}+\phi_2X_{t-2}+\cdots+\phi_pX_{t-p}+\varepsilon_t,\qquad \varepsilon_t\sim\mathrm{WN}(0,\sigma^2)
$$

오늘 값을 과거 $p$개 값의 가중합과 새 충격으로 설명한다. AR은 AutoRegressive, "자기 자신에게 회귀한다"는 뜻이다. 추정할 것은 가중치 $\phi_1,\ldots,\phi_p$와 충격의 크기 $\sigma^2$다.

### 1.4 정상성

$$
\mathrm{Cov}(X_t,X_{t-k})=\gamma_k\quad(\text{시차 }k\text{만의 함수, 시점 }t\text{와 무관}),\qquad \gamma_{-k}=\gamma_k
$$

"하루 차이 나는 값끼리의 닮음"이 1월이든 7월이든 같아야 데이터 전체에서 하나의 $\gamma_1$을 계산하는 것이 의미가 있다. 주가 **수준**은 추세가 있어 대개 정상이 아니고, 주가 **수익률**(로그 차분)은 대체로 정상에 가깝다.

---

## 2. AR 계수를 구하는 방법 네 가지

### 2.1 한눈에 보기

| | 방법 | 아이디어 | 추정 모형이 늘 정상인가 | 주로 쓰는 곳 |
|---|---|---|---|---|
| ① | **OLS** (조건부 최소제곱) | 오늘 값을 과거값에 **회귀**해 예측오차 제곱합을 최소화 | 아니다 | 회귀분석, 외부 변수 포함 모형 |
| ② | **Yule-Walker** (적률법) | 표본 **자기상관**을 이론식에 넣고 연립방정식을 푼다 | 그렇다 | PACF 계산, 빠른 탐색, 초기값 |
| ③ | **Burg** | **앞 방향과 뒤 방향** 예측오차를 함께 최소화 | 그렇다 | 신호 처리, 짧은 자료 |
| ④ | **MLE** (최대우도) | 관측 데이터가 나올 **확률(우도)** 이 가장 큰 계수 | 제약을 걸면 그렇다 | 통계 패키지 기본값(ARIMA) |

네 방법은 같은 AR 모형의 같은 계수를 구한다. 다른 것은 "무엇을 맞추느냐"뿐이다.

### 2.2 AR(1)로 보면 분자는 같고 분모만 다르다

$d_t=x_t-\bar x$(평균에서 벗어난 정도)로 쓰면, AR(1)에서 ①~③의 공식은 다음과 같다.

$$
\text{OLS}:\quad\hat\phi=\frac{\sum_{t=2}^{n}d_t\,d_{t-1}}{\sum_{t=2}^{n}d_{t-1}^2}
$$

$$
\text{YW}:\quad\hat\phi=\frac{\sum_{t=2}^{n}d_t\,d_{t-1}}{\sum_{t=1}^{n}d_t^2}
$$

$$
\text{Burg}:\quad\hat\phi=\frac{2\sum_{t=2}^{n}d_t\,d_{t-1}}{\sum_{t=2}^{n}\bigl(d_t^2+d_{t-1}^2\bigr)}
$$

- 분자는 모두 "이웃한 편차의 곱의 합"이다.
- **YW의 분모는 모든 값의 제곱합이라 가장 크다.** 그래서 YW 추정값이 가장 작게(0에 가깝게) 나온다. §4의 비교 결과가 이 한 줄로 설명된다.
- ④ MLE는 닫힌 공식이 없어 컴퓨터로 최적화한다.

### 2.3 방법 ① OLS — 회귀로 푼다

$$
\hat{\boldsymbol\phi}^{\,\mathrm{OLS}}=\arg\min_{\boldsymbol\phi}\;\sum_{t=p+1}^{n}\bigl(x_t-\phi_1x_{t-1}-\cdots-\phi_px_{t-p}\bigr)^2
$$

AR 모형을 "설명변수가 자기 과거값인 회귀식"으로 보고, 교과서의 최소제곱 해 $\hat{\boldsymbol\phi}=(X^\top X)^{-1}X^\top y$를 쓴다.

- 장점: 회귀분석을 알면 바로 쓸 수 있다. 소표본 치우침이 YW보다 작다. 외부 변수를 함께 넣기 쉽다.
- 약점: 추정 모형이 비정상으로 나올 수 있다. 처음 $p$개 관측치는 설명변수로만 쓰인다.

### 2.4 방법 ② Yule-Walker — 자기상관을 맞춘다

$$
\gamma_k=\phi_1\gamma_{k-1}+\phi_2\gamma_{k-2}+\cdots+\phi_p\gamma_{k-p},\qquad k=1,2,\ldots,p
$$

이론적으로 성립하는 이 관계식에 데이터로 계산한 자기공분산을 넣고, $\phi$에 대한 연립방정식을 푼다. 통계학 용어로는 적률법(method of moments)이다. §3에서 자세히 본다.

- 장점: 자기상관만 있으면 된다. 추정 모형이 항상 정상이다. 차수를 하나씩 올리며 모든 차수를 한 번에 푼다(PACF가 덤으로 나온다).
- 약점: 소표본에서 0 쪽으로 치우친다. 계수가 1에 가까울 때 가장 크게 빗나간다. 정상성이 깨지면 경고 없이 틀린다.

### 2.5 방법 ③ Burg — 앞으로도, 뒤로도 예측한다

$$
\min_{\phi}\;\sum_{t=2}^{n}\Bigl[\underbrace{(d_t-\phi\,d_{t-1})^2}_{\text{앞 방향 오차}}+\underbrace{(d_{t-1}-\phi\,d_t)^2}_{\text{뒤 방향 오차}}\Bigr]
$$

정상 시계열은 시간을 거꾸로 뒤집어도 성질이 같다. 그래서 "어제로 오늘을 예측하는 오차"와 "오늘로 어제를 거꾸로 예측하는 오차"를 함께 줄이면 같은 데이터에서 정보를 두 번 쓰는 셈이 된다. 고차 AR에서는 이 과정을 차수마다 반복한다.

- 장점: 추정 모형이 항상 정상이다. 짧은 자료에서도 치우침이 작다. 신호 처리의 표준 방법이다.
- 약점: 개념이 낯설다. 경제·금융 패키지의 기본값은 아니다.

### 2.6 방법 ④ MLE — 확률을 최대화한다

잡음이 정규분포를 따른다고 가정하면, 주어진 $\phi,\mu,\sigma^2$에서 지금의 데이터가 나올 확률(우도)을 계산할 수 있다. AR(1)의 로그우도는

$$
\log L(\phi,\mu,\sigma^2)=-\frac{n}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{t=2}^{n}\bigl(x_t-\mu-\phi(x_{t-1}-\mu)\bigr)^2+(\text{첫 관측치 항})
$$

이고, 이를 가장 크게 만드는 값을 고른다.

$$
\hat\phi^{\,\mathrm{MLE}}=\arg\max_{\phi,\mu,\sigma^2}\;\log L(\phi,\mu,\sigma^2\mid x_1,\ldots,x_n)
$$

- 장점: 통계적으로 가장 효율적이다. 표준오차·AIC·BIC가 함께 나온다. ARMA·ARIMA로 그대로 확장된다.
- 약점: 공식이 없어 반복 계산이 필요하다. 분포 가정이 틀리면 흔들린다. 반복 계산의 출발점으로 YW 결과를 쓰는 경우가 많다.

---

## 3. Yule-Walker 자세히

### 3.1 유도: 양변에 과거값을 곱하고 공분산을 취한다

공분산은 상수를 밖으로 뺄 수 있고($\mathrm{Cov}(aX,Y)=a\,\mathrm{Cov}(X,Y)$), 합을 나눌 수 있다. AR(2), $k\ge1$이면

$$
\gamma_k=\phi_1\underbrace{\mathrm{Cov}(X_{t-1},X_{t-k})}_{\gamma_{k-1}}+\phi_2\underbrace{\mathrm{Cov}(X_{t-2},X_{t-k})}_{\gamma_{k-2}}+\underbrace{\mathrm{Cov}(\varepsilon_t,X_{t-k})}_{0}
$$

- 가운데 두 항을 $\gamma_{k-1},\gamma_{k-2}$로 줄이는 단계가 **정상성을 쓰는 바로 그 지점**이다.
- 오늘의 충격 $\varepsilon_t$는 과거값 $X_{t-k}$와 무관하므로 마지막 항은 0이다.

일반 $p$로 쓴 것이 **Yule-Walker 방정식**이다. 식 $p$개, 미지수 $p$개라서 정확히 풀린다.

$$
\gamma_k=\phi_1\gamma_{k-1}+\phi_2\gamma_{k-2}+\cdots+\phi_p\gamma_{k-p},\qquad k=1,2,\ldots,p
$$

### 3.2 $k=0$ 식은 계수가 아니라 충격의 크기 $\sigma^2$를 준다

$k=0$이면 오늘 충격 $\varepsilon_t$가 오늘 값 $X_t$ 안에 들어 있으므로 둘의 공분산은 0이 아니라 $\sigma^2$다.

$$
\gamma_0=\phi_1\gamma_1+\phi_2\gamma_2+\sigma^2
$$

> [!warning] 영상 판서 정정 (중요)
> 영상은 이 $\sigma^2$를 0으로 지우고 $k=0,1$ 두 식으로 $\phi$를 풀었다. 그 계수 행렬 $\begin{pmatrix}\gamma_1&\gamma_2\\ \gamma_0&\gamma_1\end{pmatrix}$로 풀면 틀린 답이 나온다. 참값 $(0.60,\,0.25)$인 AR(2) 5,000개 시뮬레이션에서 영상의 계는 $(-2.36,\,3.96)$, 올바른 계($k=1,2$)는 $(0.596,\,0.253)$을 냈다. **계수는 $k=1,\ldots,p$ 식으로 풀고, $k=0$ 식은 $\sigma^2$ 계산에 쓴다.**

![[figs/fig_video_error.png]]

### 3.3 행렬로 정리

$$
\underbrace{\begin{pmatrix}\gamma_0&\gamma_1&\cdots&\gamma_{p-1}\\ \gamma_1&\gamma_0&\cdots&\gamma_{p-2}\\ \vdots&\vdots&\ddots&\vdots\\ \gamma_{p-1}&\gamma_{p-2}&\cdots&\gamma_0\end{pmatrix}}_{\Gamma_p}\begin{pmatrix}\phi_1\\ \vdots\\ \phi_p\end{pmatrix}=\underbrace{\begin{pmatrix}\gamma_1\\ \vdots\\ \gamma_p\end{pmatrix}}_{\boldsymbol\gamma_p}
$$

양변을 $\gamma_0$(분산)로 나누면 자기상관 $\rho_k=\gamma_k/\gamma_0$만으로 된 형태가 된다. 계수는 똑같이 나온다.

$$
\underbrace{\begin{pmatrix}1&\rho_1&\cdots&\rho_{p-1}\\ \rho_1&1&\cdots&\rho_{p-2}\\ \vdots&\vdots&\ddots&\vdots\\ \rho_{p-1}&\rho_{p-2}&\cdots&1\end{pmatrix}}_{R}\begin{pmatrix}\phi_1\\ \vdots\\ \phi_p\end{pmatrix}=\underbrace{\begin{pmatrix}\rho_1\\ \vdots\\ \rho_p\end{pmatrix}}_{r}
$$

### 3.4 추정 절차 3단계

**1단계 — 표본 자기공분산(자기상관) 계산**

$$
\hat\gamma_k=\frac{1}{n}\sum_{t=k+1}^{n}(x_t-\bar x)(x_{t-k}-\bar x),\qquad \hat\rho_k=\hat\gamma_k/\hat\gamma_0
$$

**2단계 — 연립방정식 풀기:** $\hat{\boldsymbol\phi}=\hat R^{-1}\hat r$

**3단계 — 충격 분산 계산** ($k=0$ 식 이용)

$$
\hat\sigma^2=\hat\gamma_0\bigl(1-\hat\phi_1\hat\rho_1-\cdots-\hat\phi_p\hat\rho_p\bigr)
$$

분모를 $n-k$가 아니라 $n$으로 두는 이유: 그래야 행렬이 항상 잘 풀리는 행렬(양정치)이 되고, 추정된 모형이 항상 정상이 된다. AR(1)이면 행렬도 필요 없다.

$$
\hat\phi_1=\hat\rho_1,\qquad \hat\sigma^2=\hat\gamma_0\bigl(1-\hat\phi_1^2\bigr)
$$

### 3.5 AR(1) 수치 예제 — 관측치 6개로 끝까지

어떤 지표를 6번 관측해 $x=(3,\,2,\,2,\,0,\,-1,\,0)$을 얻었다.

![[figs/fig_ar1_example.png]]

**1단계 — 평균을 빼 편차를 만든다.** $\bar x=6/6=1$.

| $t$ | 1 | 2 | 3 | 4 | 5 | 6 | 합 |
|---|---|---|---|---|---|---|---|
| $x_t$ | 3 | 2 | 2 | 0 | −1 | 0 | 6 |
| $d_t$ | +2 | +1 | +1 | −1 | −2 | −1 | 0 |
| $d_t^2$ | 4 | 1 | 1 | 1 | 4 | 1 | **12** |
| $d_t\,d_{t-1}$ | — | 2 | 1 | −1 | 2 | 2 | **6** |

**2단계 — 자기공분산.** 분모는 두 경우 모두 $n=6$이다.

$$
\hat\gamma_0=\frac{1}{6}(4+1+1+1+4+1)=\frac{12}{6}=2
$$

$$
\hat\gamma_1=\frac{1}{6}(2+1-1+2+2)=\frac{6}{6}=1
$$

**3단계 — 계수와 충격 분산.**

$$
\hat\rho_1=\frac{\hat\gamma_1}{\hat\gamma_0}=0.5\;\Rightarrow\;\hat\phi_1=0.5,\qquad \hat\sigma^2=2\,(1-0.5^2)=1.5
$$

**4단계 — 한 걸음 앞 예측.**

$$
\hat x_7=\bar x+\hat\phi_1(x_6-\bar x)=1+0.5\,(0-1)=0.5
$$

- $\hat\phi_1=0.5$: 평균에서 벗어난 정도가 다음 기에 **절반만 남는다**.
- $\hat\sigma^2=1.5$: 전체 분산 2 가운데 25%($\hat\rho_1^2$)는 어제 값으로 설명되고, 75%는 새 충격이다.
- 같은 데이터에 다른 방법을 쓴 결과는 §4.1에서 비교한다.

### 3.6 AR(2) 수치 예제

데이터에서 $\hat\rho_1=0.80$, $\hat\rho_2=0.55$를 얻었다고 하자.

$$
\begin{pmatrix}1&0.8\\ 0.8&1\end{pmatrix}\begin{pmatrix}\phi_1\\ \phi_2\end{pmatrix}=\begin{pmatrix}0.80\\ 0.55\end{pmatrix}
$$

$$
\hat\phi_1=\frac{\rho_1(1-\rho_2)}{1-\rho_1^2}=\frac{0.8\times0.45}{0.36}=1.00
$$

$$
\hat\phi_2=\frac{\rho_2-\rho_1^2}{1-\rho_1^2}=\frac{0.55-0.64}{0.36}=-0.25
$$

$$
\hat\sigma^2=\hat\gamma_0(1-1.00\times0.80+0.25\times0.55)=0.3375\,\hat\gamma_0
$$

어제 값의 영향은 크고(1.00), 그제 값은 약간 되돌리는 방향(−0.25)으로 작용한다. 충격 몫은 전체 분산의 약 34%다. 계수 1.00이 있어도 AR(2) 전체로는 정상이다(특성방정식 $1-z+0.25z^2=(1-0.5z)^2$의 근이 2로 1보다 크다).

### 3.7 토플리츠 행렬과 레빈슨-더빈 재귀

$\Gamma_p$의 $(i,j)$ 원소는 $\gamma_{|i-j|}$다. **모든 대각선에 같은 값**이 놓이는 이런 행렬을 **토플리츠(Toeplitz) 행렬**이라 한다. $p^2$칸이 실제로는 $p$개 값으로 채워진다.

![[figs/fig_toeplitz.png]]

이 규칙성 덕에 **레빈슨-더빈(Levinson-Durbin) 재귀**로 풀 수 있다. AR(1)을 먼저 풀고, 그 답을 조금 고쳐 AR(2)를 얻고, 다시 AR(3)을 얻는 **계단 오르기** 방식이다. $v_0=\gamma_0$에서 시작해 $k=1,\ldots,p$마다

$$
\phi_{kk}=\frac{\gamma_k-\sum_{j=1}^{k-1}\phi_{k-1,j}\,\gamma_{k-j}}{v_{k-1}}
$$

$$
\phi_{kj}=\phi_{k-1,j}-\phi_{kk}\,\phi_{k-1,k-j}\;\;(j<k),\qquad v_k=v_{k-1}(1-\phi_{kk}^2)
$$

$\phi_{kj}$는 "차수 $k$ 모형의 $j$번째 계수"다. 덤으로 얻는 것:

- $\phi_{kk}$가 시차 $k$의 **편자기상관(PACF)** 이다.
- $v_k$는 차수 $k$ 모형의 잡음 분산이며 $v_p=\hat\sigma^2$이다.
- $|\phi_{kk}|<1$이 유지되므로 추정 모형은 항상 정상이다.

### 3.8 차수 $p$ 정하기

![[figs/fig_acf_pacf.png]]

- **PACF**: AR($p$)의 편자기상관은 시차 $p$ 뒤에서 뚝 끊긴다. 위 그림(참값 AR(2))은 시차 2 뒤로 유의 한계 안에 들어간다 → $p=2$.
- **ACF**: AR 과정의 자기상관은 끊기지 않고 천천히 줄어든다.
- **정보기준(AIC·BIC)**: $p=1,2,\ldots$로 각각 추정해 가장 작은 값을 고른다. 레빈슨-더빈이 모든 차수의 $v_k$를 한 번에 주므로 비교가 쉽다.

---

## 4. 방법 비교와 선택

### 4.1 같은 데이터, 네 방법

§3.5의 6개 관측치에 네 방법을 쓰면 다음과 같다. 이웃한 편차 곱의 합은 6이다.

| 방법 | 계산 | $\hat\phi_1$ | 이유 |
|---|---|---|---|
| ① OLS | $6/(12-1)=6/11$ | 0.545 | 마지막 편차는 분모에서 빠진다 |
| ② Yule-Walker | $6/12$ | 0.500 | 분모가 가장 커서 가장 작다 |
| ③ Burg | $(2\times6)/(8+11)=12/19$ | 0.632 | 앞뒤 두 제곱합의 평균을 분모로 |
| ④ MLE | 수치 최적화(평균도 함께 추정: 1.20) | 0.662 | 평균을 따로 추정해 값이 다르다 |

관측치가 6개처럼 적으면 방법마다 답이 크게 갈린다.

### 4.2 모의실험: 언제 차이가 나는가

AR(1), 관측치 100개, 반복 500회.

![[figs/fig_methods_mc.png]]

| 참값 | OLS | YW | Burg | MLE |
|---|---|---|---|---|
| $\phi=0.5$ (정상) | 0.477 | 0.472 | 0.477 | 0.477 |
| $\phi=0.95$ (단위근 근처) | 0.907 | **0.894** | 0.906 | 0.909 |

정상 구간에서는 네 방법이 사실상 같다. 1에 가까워지면 YW만 뒤처지고, MLE가 참값에 가장 가깝다. §2.2의 "YW 분모가 가장 크다"가 그대로 드러난 결과다.

### 4.3 선택 기준

| 상황 | 권장 방법 | 이유 |
|---|---|---|
| 데이터가 길고 확실히 정상 | 아무 방법이나 (YW 포함) | 네 방법이 거의 같은 답 |
| 짧은 자료 · 계수가 1에 가까움 | Burg 또는 MLE | YW의 치우침이 가장 크다 |
| 표준오차·AIC 등 추론이 필요 | MLE | 통계량이 함께 나온다 |
| 외부 변수를 함께 넣는 회귀 | OLS | 회귀식 확장이 자연스럽다 |
| PACF 계산 · MLE 초기값 | Yule-Walker | 차수별 결과를 한 번에 준다 |

실무 기본값은 MLE, 빠른 탐색과 PACF는 Yule-Walker다.

---

## 5. 활용과 주의

### 5.1 어디에 쓰이나

| 분야 | 쓰임 | YW가 선택되는 이유 |
|---|---|---|
| 신호 처리·음성 | 선형예측부호화(LPC) — 음성 코덱이 짧은 구간마다 AR 계수를 다시 구한다 | 실시간으로 반복 계산해야 한다 |
| 스펙트럼 추정 | AR 계수로 신호의 주파수 성분을 추정 | 짧은 자료에서도 매끄러운 결과 |
| 경제·금융 시계열 | 물가·금리·수익률의 AR 모형과 예측 | 추정 모형이 늘 정상이라 예측이 발산하지 않음 |
| 분석 도구 | PACF 계산, AR 차수 선택, ARIMA 초기값 | 레빈슨-더빈이 PACF를 바로 준다 |

> [!note] 흔한 과장 바로잡기
> "MP3가 LPC를 쓴다"는 설명이 인터넷에 보이는데, MP3는 변환(MDCT) 기반 음악 코덱이다. LPC는 주로 휴대전화 음성 코덱 쪽이다. 금융 변동성 모델링도 보통은 GARCH의 영역이며, YW는 보조 용도에 그친다.

### 5.2 정상성이 깨지면

AR(1), 관측치 100개, 반복 2,000회에서 YW와 OLS를 비교했다.

| 참 과정 | 참 $\phi$ | YW 평균 | OLS 평균 |
|---|---|---|---|
| 정상 | 0.50 | 0.470 | 0.490 |
| 단위근 근처 | 0.95 | 0.895 | 0.934 |
| 랜덤워크(비정상) | 1.00 | 0.929 | 0.982 |

![[figs/fig_mc.png]]

랜덤워크(주가 수준처럼 한 방향으로 떠도는 과정)를 넣어도 YW는 $\hat\phi<1$을 돌려주며, 경고 없이 "평균으로 되돌아오는 시계열"이라고 답한다. YW 모형은 구조적으로 정상 영역 안에 갇혀 있기 때문이다. 영상의 "비정상이면 쓰레기 값"은 과장이지만 방향은 맞다.

실무 순서:

1. 그래프를 그리고 ADF·KPSS 검정으로 정상성을 확인한다. 가격 수준이면 수익률(로그 차분)로 바꾼다.
2. 정상이면 원하는 방법으로 추정한다. 짧거나 1에 가까우면 Burg·MLE.
3. 여러 방법의 결과를 나란히 놓고, 크게 다르면 원인을 찾는다.

### 5.3 코드

```python
from statsmodels.regression.linear_model import yule_walker, burg
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

phi_ols   = AutoReg(x, lags=p).fit().params          # ① OLS (상수항 포함)
phi_yw, _ = yule_walker(x, order=p, method="mle")   # ② Yule-Walker (분모 n)
phi_bg, _ = burg(x, order=p)                          # ③ Burg
phi_ml    = ARIMA(x, order=(p, 0, 0)).fit().params    # ④ MLE (상수·σ² 포함)
```

`yule_walker`의 기본값은 `method="adjusted"`(분모 $n-k$)이므로, 추정 모형의 정상성을 보장하려면 `method="mle"`를 지정한다. Yule-Walker를 직접 구현하면 다음과 같다.

```python
import numpy as np
from scipy.linalg import solve_toeplitz

def yule_walker_np(x, p):
    x = x - x.mean(); n = len(x)
    g = np.array([x[:n-k] @ x[k:] / n for k in range(p + 1)])   # 1단계: 자기공분산
    phi = solve_toeplitz(g[:p], g[1:p+1])                        # 2단계: 레빈슨-더빈
    sigma2 = g[0] - phi @ g[1:p+1]                               # 3단계: 충격 분산
    return phi, sigma2
```

---

## 6. 정리

| 질문 | 답 |
|---|---|
| AR 계수를 구하는 방법은 | ① OLS ② Yule-Walker ③ Burg ④ MLE |
| Yule-Walker는 무엇을 푸나 | 자기상관으로 만든 연립방정식 $R\boldsymbol\phi=r$ |
| 무엇을 가정하나 | 정상성: $\gamma_k$가 시점 $t$에 무관 |
| 방법 간 차이는 언제 생기나 | 자료가 짧거나 계수가 1에 가까울 때. 이때 YW가 가장 작게 추정 |
| YW가 덤으로 주는 것 | PACF, 충격 분산 $\hat\sigma^2$, 추정 모형의 정상성 보장 |
| 차수는 | PACF가 끊기는 곳, 또는 AIC·BIC 최소 |

### 확인 문제

1. AR(1)에서 $\hat\rho_1=0.6$, $\hat\gamma_0=2$이다. YW 추정값 $\hat\phi_1$과 $\hat\sigma^2$는? (답: $0.6$, $2(1-0.36)=1.28$)
2. Yule-Walker 방정식을 유도할 때 정상성은 어느 단계에서 쓰였나? (답: $\mathrm{Cov}(X_{t-1},X_{t-k})$를 $\gamma_{k-1}$로 바꾸는 단계)
3. $k=0$ 식을 계수 추정에 쓰면 안 되는 이유는? (답: $\mathrm{Cov}(\varepsilon_t,X_t)=\sigma^2$라는 미지수가 하나 더 들어 있기 때문)
4. 관측치가 적고 계수가 1에 가까워 보인다. 어떤 방법을 쓰겠나? (답: Burg 또는 MLE. YW는 분모가 가장 커서 0 쪽으로 치우친다)

---

## 부록 A. 계산 속도 — OLS와 Yule-Walker

> 관심 있는 수강생을 위한 보충이다. 영상의 후반부가 이 내용을 다룬다. 본문 이해에는 필요 없다.

### A.1 Big-O 표기

$O(n)$은 데이터가 2배면 계산도 약 2배, $O(p^2)$는 약 4배, $O(p^3)$는 약 8배라는 뜻이다. 상수는 무시하고 늘어나는 속도만 본다. $n$은 관측치 수, $p$는 차수다.

### A.2 OLS의 계산 비용

| 단계 | 연산 | 복잡도 |
|---|---|---|
| $X^\top X$ | $(p\times n)(n\times p)$ — 결과 원소 $p^2$개 × 원소당 곱셈 $n$번 | $O(np^2)$ |
| 역행렬 | 일반 $p\times p$ 행렬 | $O(p^3)$ |
| $X^\top y$, 마지막 곱 | $(p\times n)(n\times1)$, $(p\times p)(p\times1)$ | $O(np)+O(p^2)$ |
| **합계** | $n\gg p$ | $O(np^2+p^3)$ |

> [!warning] 영상 판서 정정
> 영상은 $X^\top X$를 $O(n^2p)$로 적었으나 $O(np^2)$가 맞다. 결론(YW가 더 가볍다)은 변하지 않는다.

### A.3 Yule-Walker의 계산 비용

| 방법 | 자료 처리 | 연립방정식 풀이 | 합계 |
|---|---|---|---|
| OLS | $X^\top X$: $O(np^2)$ | 일반 역행렬: $O(p^3)$ | $O(np^2+p^3)$ |
| Yule-Walker | $\hat\gamma_0..\hat\gamma_p$: $O(np)$ | 레빈슨-더빈: $O(p^2)$ | $O(np+p^2)$ |

토플리츠 구조 덕에 레빈슨-더빈은 한 계단에 $O(k)$, 전체 $O(p^2)$로 끝난다. 자기공분산은 FFT로 $O(n\log n)$까지 낮출 수 있다.

![[figs/fig_pipeline.png]]

### A.4 실측

Python(numpy·scipy), $n=200{,}000$: $p=5$에서 12배, $p=50$에서 51배, $p=200$에서 107배 빨랐다. 절대 시간은 컴퓨터마다 다르지만 기울기 차이가 요점이다. 차수가 크고 반복 추정이 많은 음성 신호 처리 같은 분야에서 이 차이가 의미를 가진다.

![[figs/fig_timing.png]]
