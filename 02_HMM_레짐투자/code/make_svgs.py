import cairosvg, math
F = "font-family='Noto Sans CJK KR, Noto Sans CJK JP, sans-serif'"
INK, BLUE, NAVY, GRN, RED, GRY, HAIR, AMB = "#1E293B", "#2563EB", "#1E4E8C", "#2E7D4F", "#C0392B", "#64748B", "#CBD5E1", "#B7791F"
W, H = 1500, 460
DEFS = f"""<defs><marker id='ar' markerWidth='10' markerHeight='10' refX='8' refY='5' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='{BLUE}'/></marker>
<marker id='ag' markerWidth='10' markerHeight='10' refX='8' refY='5' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='{GRY}'/></marker></defs>"""
def T(x, y, s, size=32, fill=INK, bold=False, anchor="middle"):
    fw = " font-weight='bold'" if bold else ""
    return f"<text x='{x}' y='{y}' {F} font-size='{size}' fill='{fill}' text-anchor='{anchor}'{fw}>{s}</text>"
def out(body, name, h=H):
    s = f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{h}' viewBox='0 0 {W} {h}'><rect width='{W}' height='{h}' fill='white'/>{DEFS}{body}</svg>"
    open(f"fig/{name}.svg", "w").write(s); cairosvg.svg2png(bytestring=s.encode(), write_to=f"fig/{name}.png", output_width=W * 2)

# A. 정보집합: 필터링 vs 스무딩 vs 비터비
b = ""; x0, x1, xt = 330, 1440, 1000
b += T(xt, 36, "오늘 t (매매 결정 시점)", 30, BLUE, True)
b += f"<line x1='{xt}' y1='50' x2='{xt}' y2='440' stroke='{BLUE}' stroke-width='3' stroke-dasharray='8 6'/>"
rows = [("필터링", "P(sₜ | r₁…rₜ)", False, GRN, "매매에 사용 가능"), ("스무딩", "P(sₜ | r₁…r_T)", True, RED, "미래 정보 누출"), ("비터비 경로", "argmax P(s₁…s_T | r)", True, RED, "미래 정보 누출")]
for i, (n, f, fut, c, note) in enumerate(rows):
    y = 110 + i * 120
    b += T(24, y + 12, n, 34, INK, True, "start")
    b += f"<rect x='{x0}' y='{y-18}' width='{xt-x0}' height='36' rx='8' fill='{GRN}' opacity='0.85'/>"
    if fut: b += f"<rect x='{xt}' y='{y-18}' width='{x1-xt}' height='36' rx='8' fill='{RED}' opacity='0.85'/>"
    else: b += f"<rect x='{xt}' y='{y-18}' width='{x1-xt}' height='36' rx='8' fill='none' stroke='{HAIR}' stroke-width='2' stroke-dasharray='6 6'/>"
    b += T((x0 + xt) / 2, y + 11, "과거·현재 수익률", 28, "white", True)
    b += T((xt + x1) / 2, y + 11, "미래 수익률" if fut else "사용하지 않음", 28, "white" if fut else GRY, True)
    b += T(24, y + 55, note, 28, c, anchor="start")
out(b, "svg_infoset")

# B. 계보 로드맵 (지그재그)
nodes = [("1989", "Hamilton", ["마르코프 전환", "경기순환 모형"]), ("2002", "Ang·Bekaert", ["레짐 전환", "국제 자산배분"]),
         ("2007", "Guidolin·Timmermann", ["4-레짐", "다자산 배분"]), ("2012", "Kritzman 외", ["난기류·인플레·", "성장 레짐"]),
         ("2020", "Nystrup 외", ["점프 벌점으로", "지속성 확보"]), ("2024", "Shu·Yu·Mulvey", ["점프모형", "하방위험 축소"]),
         ("2025–26", "최신 확장", ["퍼지 점프모형·", "Wasserstein HMM"])]
b = ""; H2 = 560; yA = 280
b += f"<line x1='40' y1='{yA}' x2='1460' y2='{yA}' stroke='{HAIR}' stroke-width='6'/>"
for i, (yr, who, desc) in enumerate(nodes):
    x = 110 + i * 213; up = i % 2 == 0; c = BLUE if i < 4 else (GRN if i < 6 else AMB)
    b += f"<circle cx='{x}' cy='{yA}' r='16' fill='{c}'/>"
    by = 40 if up else 330; ly1, ly2 = (yA - 16, by + 190) if up else (yA + 16, by)
    b += f"<line x1='{x}' y1='{ly1}' x2='{x}' y2='{ly2}' stroke='{c}' stroke-width='3'/>"
    bx = min(max(x - 150, 10), W - 310)
    b += f"<rect x='{bx}' y='{by}' width='300' height='190' rx='14' fill='white' stroke='{c}' stroke-width='3'/>"
    b += T(bx + 150, by + 42, yr, 32, c, True) + T(bx + 150, by + 86, who, 25 if len(who) > 14 else 28, INK, True)
    b += T(bx + 150, by + 130, desc[0], 27, GRY) + T(bx + 150, by + 166, desc[1], 27, GRY)
out(b, "svg_lineage", H2)

# C. 실무 설계 파이프라인
steps = [("1", "특징변수", "하방변동성", "소르티노·VIX"), ("2", "모형 선택", "K는 BIC,", "λ는 교차검증"),
         ("3", "추론", "필터링만 사용", "라벨 고정"), ("4", "신호 규칙", "밴드·확률가중", "전환 억제"),
         ("5", "실행", "1일 지연", "거래비용 차감"), ("6", "평가", "워크포워드", "다중검정 보정")]
b = ""; BW = 222; G = 30; x0 = 24
for i, (no, t, d1, d2) in enumerate(steps):
    x = x0 + i * (BW + G); last = i == 5
    fill = NAVY if last else "white"; tc = "white" if last else INK
    b += f"<rect x='{x}' y='20' width='{BW}' height='340' rx='16' fill='{fill}' stroke='{NAVY if last else BLUE}' stroke-width='3'/>"
    b += f"<circle cx='{x+BW/2}' cy='80' r='36' fill='{'white' if last else BLUE}'/>"
    b += T(x + BW / 2, 94, no, 38, NAVY if last else "white", True) + T(x + BW / 2, 175, t, 36, tc, True)
    b += T(x + BW / 2, 250, d1, 30, tc if last else GRY) + T(x + BW / 2, 295, d2, 30, tc if last else GRY)
    if i < 5: b += f"<line x1='{x+BW+2}' y1='190' x2='{x+BW+G-3}' y2='190' stroke='{BLUE}' stroke-width='5' marker-end='url(#ar)'/>"
b += T(750, 425, "모형보다 정보집합·지연·비용·평가 설계가 성과를 더 크게 좌우한다", 32, NAVY, True)
out(b, "svg_design")
print("ok")
