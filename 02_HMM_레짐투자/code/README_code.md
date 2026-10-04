# HMM 레짐 트레이딩 확장판 — 재현 코드

데이터는 yfinance로 직접 내려받습니다(별도 데이터 파일 없음). 실행 순서:

1. 데이터 다운로드: TSLA, SPY, ^VIX, ^IRX, 대형주 40종목 (e1·e2·e3 스크립트 상단 참고, `data/` 폴더에 CSV 저장)
2. `e1_tsla.py` — TSLA 3-상태 HMM, 전체 표본과 1년 창 초기값 민감도
3. `e4_bic.py` — K = 1…5 BIC 비교
4. `e2_xsec.py` → `a_e2.py` — 40종목 워크포워드 재현(필터링, 월 1회 재적합), 무작위 1,000경로 비교
5. `e3_regime.py` → `a_e3.py` — SPY 0/1 레짐 전략: HMM vs 통계적 점프모형(λ = 0·10·50·100), 1일 지연·편도 10bp
6. `f_*.py`, `make_svgs.py`, `render_eqs.py` — 그림·수식

필요 패키지: `pip install yfinance hmmlearn statsmodels cairosvg matplotlib scipy pandas`
주의: 멀티프로세싱 안에서 hmmlearn 기본 초기화(KMeans)가 멈출 수 있어 순차 실행과 수동 초기화를 사용했습니다.
