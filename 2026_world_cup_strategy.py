#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
#  World Cup 2026 — 全赛事 Monte Carlo 模型 + 分层市场自动校准 + EV/Kelly
# =====================================================================
#  设计目标:你只改两类东西 ——
#     (1) TEAMS         : 48 队的 elo / group / inj
#     (2) 各市场赔率字典 : 你手上有哪个就贴哪个(冠军/出线/八强/四强/单场)
#  然后跑 calibrate_all():每个市场自动拟合到它对应的【唯一】参数,
#     单场赔率 → SIGMA      (单场不可预测性)
#     出线赔率 → GOAL_DIV   (小组赛进球模型的灵敏度)
#     八强/四强/冠军 → ELO_SCALE (整体强弱离散度 / 赛会方差)
#  ——改最少的参数,拿最大的效果。
#
#  流程: python3 wc2026_model.py            # 用当前参数出报告
#         python3 wc2026_model.py calibrate  # 先 calibrate_all 再出报告
#         python3 wc2026_model.py brier      # SIGMA 的 Brier 自检
# =====================================================================
import sys, math
import numpy as np

# ------------------------- 全局参数(会被校准覆盖) -------------------------
N_SIMS         = 80_000    # 报告用的模拟次数(精度↑则调大;SE≈√(p(1-p)/N))
SIGMA          = 85        # 每场 form/运气 波动(Elo点)。由单场赔率/Brier 校准
ELO_SCALE      = 0.90      # Elo 围绕均值的离散度缩放(<1 拉平大热)。由深度市场校准
GOAL_DIV       = 165       # 小组赛 Poisson:多少 Elo 差≈1 球净胜。由出线赔率校准
BASE_GOALS     = 2.60      # 小组赛单场总进球基准(历史 WC ~2.5-2.8)
HOST_ADV       = 35        # 东道主(美/墨/加)主场 Elo 加成
KELLY_FRACTION = 0.25      # 分数 Kelly(¼ 稳健;长尾建议更小)
MARKET_BLEND   = 0.60      # fair = 市场×BLEND + 模型×(1-BLEND)
SEED           = 7         # 随机种子

# =====================================================================
#  (1) 球队:  名称 -> dict(elo=基础Elo, group=小组A..L, inj=伤病Elo调整)
# ---------------------------------------------------------------------
#  数据来源/怎么改:
#   - elo : eloratings.net(World Football Elo,由历史比分+K因子+净胜球算)。
#           或把 FIFA 积分线性映射。每场比赛后会变,赛前刷新一次即可。
#   - group: 抽签结果(固定),来自 FIFA 官方分组。
#   - inj : 伤病/缺阵的 Elo 折价(负=变弱)。最干净的测法见 README 注释:
#           伤病新闻出来后看该队赔率漂移,两端去水反解 Elo delta。
#           粗略量级:世界级核心 30-70;主力 15-30;轮换 5-15;
#           概率性伤(可能赶上)按"出场概率×满额折价"打折。
#  (host=True 的队会自动加 HOST_ADV;在 _is_host 里按名字判断)
# =====================================================================
TEAMS = {
 # 组 A
 "Mexico":dict(elo=1850,group="A",inj=0), "South Korea":dict(elo=1790,group="A",inj=0),
 "South Africa":dict(elo=1700,group="A",inj=0), "Czechia":dict(elo=1780,group="A",inj=0),
 # 组 B
 "Canada":dict(elo=1800,group="B",inj=0), "Switzerland":dict(elo=1830,group="B",inj=0),
 "Qatar":dict(elo=1690,group="B",inj=0), "Bosnia":dict(elo=1770,group="B",inj=0),
 # 组 C
 "Brazil":dict(elo=1905,group="C",inj=-45), "Morocco":dict(elo=1875,group="C",inj=0),
 "Scotland":dict(elo=1770,group="C",inj=0), "Haiti":dict(elo=1500,group="C",inj=0),
 # 组 D
 "USA":dict(elo=1830,group="D",inj=0), "Paraguay":dict(elo=1750,group="D",inj=0),
 "Australia":dict(elo=1740,group="D",inj=0), "Turkiye":dict(elo=1820,group="D",inj=0),
 # 组 E
 "Germany":dict(elo=1910,group="E",inj=-10), "Ecuador":dict(elo=1800,group="E",inj=0),
 "Ivory Coast":dict(elo=1780,group="E",inj=0), "Curacao":dict(elo=1550,group="E",inj=0),
 # 组 F
 "Netherlands":dict(elo=1945,group="F",inj=0), "Japan":dict(elo=1840,group="F",inj=0),
 "Tunisia":dict(elo=1740,group="F",inj=0), "Sweden":dict(elo=1810,group="F",inj=0),
 # 组 G
 "Belgium":dict(elo=1890,group="G",inj=0), "Iran":dict(elo=1760,group="G",inj=0),
 "Egypt":dict(elo=1780,group="G",inj=0), "New Zealand":dict(elo=1620,group="G",inj=0),
 # 组 H
 "Spain":dict(elo=2060,group="H",inj=-8), "Uruguay":dict(elo=1885,group="H",inj=0),
 "Saudi Arabia":dict(elo=1680,group="H",inj=0), "Cape Verde":dict(elo=1640,group="H",inj=0),
 # 组 I (死亡之组)
 "France":dict(elo=2045,group="I",inj=-10), "Senegal":dict(elo=1880,group="I",inj=0),
 "Norway":dict(elo=1845,group="I",inj=0), "Iraq":dict(elo=1675,group="I",inj=0),
 # 组 J
 "Argentina":dict(elo=1990,group="J",inj=-20), "Austria":dict(elo=1810,group="J",inj=0),
 "Algeria":dict(elo=1780,group="J",inj=0), "Jordan":dict(elo=1660,group="J",inj=0),
 # 组 K
 "Portugal":dict(elo=1955,group="K",inj=0), "Colombia":dict(elo=1870,group="K",inj=0),
 "Uzbekistan":dict(elo=1720,group="K",inj=0), "DR Congo":dict(elo=1730,group="K",inj=0),
 # 组 L
 "England":dict(elo=1990,group="L",inj=0), "Croatia":dict(elo=1855,group="L",inj=0),
 "Panama":dict(elo=1680,group="L",inj=0), "Ghana":dict(elo=1750,group="L",inj=0),
}
HOSTS = {"USA", "Mexico", "Canada"}            # 自动获得 HOST_ADV

# 小组 -> 半区('L'/'R')。决定淘汰赛对阵结构(不是数值,改这里=改对阵树)。
# 依据抽签:Spain(H)/France(I) 在左侧死亡区;England(L)/Argentina(J)/
# Portugal(K)/Brazil(C) 在右侧。其余按 FOX 对阵树推断;有官方 R32 图就替换。
GROUP_HALF = {"E":"L","F":"L","G":"L","H":"L","I":"L","B":"L",   # 左 6 组
              "A":"R","C":"R","D":"R","J":"R","K":"R","L":"R"}   # 右 6 组

# =====================================================================
#  (2) 市场赔率(贴你手上有的;空字典=跳过对应校准)
# ---------------------------------------------------------------------
#  全部支持两种填法:美式赔率(整数)或已归一的概率(0-1 浮点)。
#  _to_prob() 会自动识别并把美式赔率转概率;概率会在使用时按需重归一。
# =====================================================================

# --- 冠军:全 48 队去水概率。来源:Polymarket/Kalshi 的 yes 价(近零水、已归一),
#     或某书【全 48 队】赔率 1/decimal 后整体归一(别只拿前几名归一!)。
CHAMP_DEVIG = {
 "Spain":.160,"France":.125,"England":.120,"Argentina":.090,"Brazil":.085,
 "Portugal":.065,"Germany":.055,"Netherlands":.035,"Norway":.030,"Belgium":.020,
 "USA":.017,"Colombia":.017,"Japan":.016,"Morocco":.016,"Uruguay":.014,
 "Croatia":.012,"Switzerland":.012,"Mexico":.011,"Ecuador":.009,"Senegal":.009,
 "Turkiye":.007,"South Korea":.005,"Canada":.005,"Sweden":.005,"Austria":.005,
 "Egypt":.0045,"Ivory Coast":.0045,"Scotland":.004,"Australia":.0035,"Algeria":.0035,
 "Czechia":.0035,"Paraguay":.003,"Iran":.003,"Bosnia":.003,"Tunisia":.0025,
 "Saudi Arabia":.002,"Uzbekistan":.002,"DR Congo":.002,"Qatar":.002,"Ghana":.002,
 "South Africa":.0015,"Panama":.001,"Cape Verde":.001,"New Zealand":.001,"Iraq":.001,
 "Jordan":.001,"Haiti":.0005,"Curacao":.0005,
}
# --- 出线(小组前二+最佳第三都算晋级)。来源:各书 "to qualify from group" 盘。
#     贴美式赔率或概率;只贴你关心的队即可。空=跳过 GOAL_DIV 校准。
GROUP_ADV_ODDS = {
 # 例: "England": -300, "Croatia": +120, ...(留空则不校准小组赛)
}
# --- 八强(reach quarter-final / 进入最后 8 强)。来源:"to reach QF" 盘。
REACH_QF_ODDS = {
 # 例: "Spain": +110, "Brazil": +160, ...
}
# --- 四强(reach semi-final / 进入最后 4 强)。来源:"to reach SF" 盘。
REACH_SF_ODDS = {
 # 例: "Spain": +260, "France": +320, ...
}
# --- 单场对阵胜负(用于校准 SIGMA)。格式: (队A, 队B, A方美式赔率或胜率).
#     来源:任意一场的让球前 H2H 三式中的"非平局两端"——最好先把平局赔率
#     按比例摊掉(de-vig)再取 A 胜的两端概率。空=跳过(改用 brier_*)。
MATCH_ODDS = [
 # 例: ("Spain","Uruguay", -180), ("France","Senegal", -150),
]

# =====================================================================
#  引擎
# =====================================================================
rng = np.random.default_rng(SEED)                     # 全局随机源
GROUPS = sorted({TEAMS[t]["group"] for t in TEAMS})   # ['A'..'L']

_MEAN = None
def _mean_elo():
    """所有真实队的平均 Elo(ELO_SCALE 围绕它缩放)。惰性缓存。"""
    global _MEAN
    if _MEAN is None:
        vals = [TEAMS[t]["elo"] for t in TEAMS]
        _MEAN = sum(vals) / len(vals)
    return _MEAN

def _is_host(t):
    return t in HOSTS

def eff_elo(t):
    """有效 Elo = 均值 + ELO_SCALE×(基础+伤病-均值) + 东道主加成。"""
    raw = TEAMS[t]["elo"] + TEAMS[t]["inj"]            # 基础+伤病
    e = _mean_elo() + ELO_SCALE * (raw - _mean_elo())  # 离散度缩放
    if _is_host(t):                                    # 东道主主场
        e += HOST_ADV
    return e

def raw_elo(t):
    """不含 ELO_SCALE 的 Elo(基础+伤病+东道主)。供 SIGMA 校准用,与 ELO_SCALE 解耦。"""
    e = TEAMS[t]["elo"] + TEAMS[t]["inj"]
    if _is_host(t): e += HOST_ADV
    return e

def p_a_beats_b(t, u, sigma=None, use_raw=False):
    """单场 A 胜 B 的【边际】概率(对 form 噪声取期望,数值积分)。
       use_raw=True 用 raw_elo(与 ELO_SCALE 解耦)——校准 SIGMA 时用。"""
    s = SIGMA if sigma is None else sigma
    ef = raw_elo if use_raw else eff_elo
    d = ef(t) - ef(u)                                  # Elo 差
    if s <= 0:
        return 1.0 / (1.0 + 10 ** (-d / 400))          # 无噪声:纯 logistic
    # n ~ N(0, √2·s):两队各加一个 N(0,s) 噪声,差的方差=2s²
    xs = np.linspace(-4, 4, 41) * (math.sqrt(2) * s)   # 积分网格(±4σ)
    w = np.exp(-(xs ** 2) / (2 * (math.sqrt(2) * s) ** 2))  # 高斯权重
    w /= w.sum()
    p = 1.0 / (1.0 + 10 ** (-(d + xs) / 400))          # 每个噪声点的胜率
    return float((p * w).sum())                        # 加权平均=边际胜率

def match(a, b):
    """抽样一场比赛(含 form 噪声),返回胜者。淘汰赛加时点球折叠进同一胜率。"""
    ea = eff_elo(a) + rng.normal(0, SIGMA)
    eb = eff_elo(b) + rng.normal(0, SIGMA)
    return a if rng.random() < 1.0 / (1.0 + 10 ** (-(ea - eb) / 400)) else b

def _goals(ea, eb):
    """小组赛单场进球(独立 Poisson)。supremacy 由 Elo 差/GOAL_DIV 决定。"""
    sup = (ea - eb) / GOAL_DIV                          # 净胜球期望
    la = max(0.12, (BASE_GOALS + sup) / 2)             # A 的进球率
    lb = max(0.12, (BASE_GOALS - sup) / 2)             # B 的进球率
    return rng.poisson(la), rng.poisson(lb)

def sim_groups():
    """模拟 12 个小组循环赛,返回 (各组排名, 全部第三名带成绩)。"""
    rank_by_group = {}                                  # 组 -> [第1,第2,第3,第4]
    thirds = []                                         # [(pts,gd,gf,队名),...]
    for g in GROUPS:
        teams = [t for t in TEAMS if TEAMS[t]["group"] == g]
        pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}; gf = {t: 0 for t in teams}
        for i in range(len(teams)):                     # 双循环打满 6 场
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                ga, gb = _goals(eff_elo(a), eff_elo(b))
                gf[a] += ga; gf[b] += gb
                gd[a] += ga - gb; gd[b] += gb - ga
                if ga > gb: pts[a] += 3
                elif gb > ga: pts[b] += 3
                else: pts[a] += 1; pts[b] += 1
        order = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t], rng.random()), reverse=True)
        rank_by_group[g] = order
        t3 = order[2]                                   # 该组第三名
        thirds.append((pts[t3], gd[t3], gf[t3], t3))
    return rank_by_group, thirds

# 16 队半区的标准蛇形种子顺序(0-indexed),保证 1、2 号种子最晚相遇
SEED16 = [0, 15, 7, 8, 4, 11, 3, 12, 2, 13, 5, 10, 6, 9, 1, 14]

def _play_half(teams16, depth):
    """打完一个 16 队半区,返回该侧决赛代表。
       同时在 depth[队]=该队赢下的淘汰赛轮数 上累加(用于分层概率)。"""
    order = sorted(teams16, key=eff_elo, reverse=True)  # 按强度定种子
    while len(order) < 16:                              # 不足 16 用极弱占位(=bye)
        order.append("__BYE__")
    bracket = [order[i] for i in SEED16]                # 排进 16 位
    won = {t: 0 for t in teams16}                       # 本半区每队赢的轮数
    while len(bracket) > 1:                             # R32→R16→QF→SF
        nxt = []
        for i in range(0, len(bracket), 2):
            a, b = bracket[i], bracket[i + 1]
            if a == "__BYE__": w = b
            elif b == "__BYE__": w = a
            else: w = match(a, b)
            if w != "__BYE__": won[w] = won.get(w, 0) + 1
            nxt.append(w)
        bracket = nxt
    for t, c in won.items():                            # 写回全局深度
        if t in depth: depth[t] = max(depth[t], c)
    return bracket[0]

def simulate(n=N_SIMS, seed=None, want_pairs=True):
    """完整赛事模拟 n 次。返回各阶段概率字典:
       adv(出线) / qf(八强) / sf(四强) / final(进决赛) / champ(夺冠) / pairs(决赛对阵)。
       说明:KO 赢 ≥1 轮=进 R16, ≥2=八强, ≥3=四强, ≥4=进决赛, ==5=夺冠。"""
    global rng
    if seed is not None:
        rng = np.random.default_rng(seed)              # common random numbers
    teams = list(TEAMS)
    adv = {t: 0 for t in teams}; qf = {t: 0 for t in teams}
    sf = {t: 0 for t in teams};  fin = {t: 0 for t in teams}
    champ = {t: 0 for t in teams}; pairs = {}
    for _ in range(n):
        rank, thirds = sim_groups()                    # 小组赛
        best_thirds = [x[3] for x in sorted(thirds, reverse=True)[:8]]  # 最佳 8 个第三
        quals = []                                     # 32 强名单
        for g in GROUPS:
            quals += rank[g][:2]                       # 前二直接出线
        quals += best_thirds                           # +8 个最佳第三
        for t in quals: adv[t] += 1                    # 记"出线"
        # 按 GROUP_HALF 把 32 强分到两个半区
        L = [t for t in quals if GROUP_HALF[TEAMS[t]["group"]] == "L"]
        R = [t for t in quals if GROUP_HALF[TEAMS[t]["group"]] == "R"]
        # 强行各 16:谁多了就把最弱的挪给对面(近似;有官方图可精确化)
        while len(L) > 16: R.append(min(L, key=eff_elo)); L.remove(min(L, key=eff_elo))
        while len(R) > 16: L.append(min(R, key=eff_elo)); R.remove(min(R, key=eff_elo))
        depth = {t: 0 for t in quals}                  # 每队 KO 赢的轮数
        wl = _play_half(L, depth); wr = _play_half(R, depth)  # 两半区决出决赛双方
        for t, c in depth.items():                     # 分层累加
            if c >= 2: qf[t] += 1                       # 八强
            if c >= 3: sf[t] += 1                       # 四强
            if c >= 4: fin[t] += 1                      # 进决赛
        w = match(wl, wr); champ[w] += 1                # 决赛
        if want_pairs:
            k = tuple(sorted((wl, wr))); pairs[k] = pairs.get(k, 0) + 1
    nrm = lambda d: {k: v / n for k, v in d.items()}
    return dict(adv=nrm(adv), qf=nrm(qf), sf=nrm(sf),
                final=nrm(fin), champ=nrm(champ),
                pairs={k: v / n for k, v in pairs.items()})

# =====================================================================
#  通用:赔率/概率工具
# =====================================================================
def american_to_decimal(a): return 1 + (a / 100 if a > 0 else 100 / -a)
def _to_prob(v):
    """识别输入:|v|>=1 当成美式赔率转概率;否则当成已是概率。"""
    if isinstance(v, (int, float)) and abs(v) >= 1.0:
        return 1.0 / american_to_decimal(v)
    return float(v)
def ev_per_dollar(p, a): return p * american_to_decimal(a) - 1
def kelly(p, a, frac=KELLY_FRACTION):
    b = american_to_decimal(a) - 1
    return max(0.0, (p * b - (1 - p)) / b) * frac

# =====================================================================
#  校准:每个市场拟合它对应的【一个】参数(网格搜索 + CRN + 排除主观edge)
# =====================================================================
def _exclude_thesis(target_keys, min_abs_inj=25):
    """校准目标剔除有强主观调整(|inj|>=阈值)的队,保住你刻意下的 edge。"""
    return {t for t in target_keys if abs(TEAMS.get(t, {}).get("inj", 0)) >= min_abs_inj}

def _kl(target_prob, model_prob, keys):
    """KL(market||model),在 keys 子集上各自重归一后比较(越小越贴合)。"""
    ms = sum(target_prob[t] for t in keys) or 1e-9
    qs = sum(model_prob.get(t, 0) for t in keys) or 1e-9
    L = 0.0
    for t in keys:
        p = target_prob[t] / ms
        q = max(model_prob.get(t, 0) / qs, 1e-6)
        L += p * math.log(p / q)
    return L

def calibrate_match_odds(matches=None, grid=None, apply=True, verbose=True):
    """【单场赔率 → SIGMA】用一批 H2H 赔率拟合单场波动(用 RAW Elo,与 ELO_SCALE 解耦)。
       matches: [(A,B, A方美式赔率或胜率), ...];默认读 MATCH_ODDS。
       最小化 Σ(模型边际胜率 - 市场胜率)²。
       注意:SIGMA(单场方差)与 ELO_SCALE(整体形状)部分冗余——
             正确做法是先用本函数(或 brier_*)用 RAW Elo 把 SIGMA 钉死,
             再用深度市场拟合 ELO_SCALE 去吸收"市场偏好/赛会方差"的残差。"""
    global SIGMA
    matches = matches or MATCH_ODDS
    if not matches:
        if verbose: print("[match] 无单场赔率,跳过(可用 brier_calibrate_sigma)。")
        return None
    grid = grid or [40, 55, 70, 85, 100, 115, 130, 150, 175, 200]
    tgt = [(a, b, _to_prob(v)) for a, b, v in matches]
    best = (None, 1e18)
    for s in grid:
        sse = sum((p_a_beats_b(a, b, sigma=s, use_raw=True) - y) ** 2 for a, b, y in tgt)
        if verbose: print(f"  [match] SIGMA={s:4d}  SSE={sse:.4f}")
        if sse < best[1]: best = (s, sse)
    if apply: SIGMA = best[0]
    edge = "(边界!扩grid)" if best[0] in (grid[0], grid[-1]) else ""
    if verbose: print(f"  >>> SIGMA={best[0]}  (单场赔率拟合, RAW Elo){edge}")
    return best[0]

def calibrate_group(target=None, grid=None, n=12_000, fit_seed=99,
                    apply=True, verbose=True):
    """【出线赔率 → GOAL_DIV】拟合小组赛进球模型灵敏度。
       target: {队:出线赔率或概率}。GOAL_DIV 越小=强队净胜球越大=越易出线。"""
    global GOAL_DIV
    target = {k: _to_prob(v) for k, v in (target or GROUP_ADV_ODDS).items()}
    if not target:
        if verbose: print("[group] 无出线赔率,跳过 GOAL_DIV 校准。")
        return None
    grid = grid or [120, 140, 160, 180, 200, 230]
    keys = [t for t in target if t in TEAMS]
    best = (None, 1e18)
    for gd in grid:
        GOAL_DIV = gd
        res = simulate(n=n, seed=fit_seed, want_pairs=False)  # CRN
        L = _kl(target, res["adv"], keys)
        if verbose: print(f"  [group] GOAL_DIV={gd:4d}  KL={L:.4f}")
        if L < best[1]: best = (gd, L)
    if apply: GOAL_DIV = best[0]
    if verbose: print(f"  >>> GOAL_DIV={best[0]}  (出线赔率拟合)")
    return best[0]

def calibrate_depth(stage="champ", target=None, grid=None, fixed_sigma=None,
                    n=15_000, fit_seed=123, apply=True, exclude_thesis=True,
                    verbose=True):
    """【八强/四强/冠军 → ELO_SCALE】用某层深度市场拟合整体离散度。
       stage ∈ {'qf','sf','final','champ'};target 缺省读对应市场字典。
       ELO_SCALE<1 拉平大热(单旋钮,可识别;SIGMA 固定)。"""
    global ELO_SCALE, SIGMA
    src = {"qf": REACH_QF_ODDS, "sf": REACH_SF_ODDS,
           "final": {}, "champ": CHAMP_DEVIG}[stage]
    target = {k: _to_prob(v) for k, v in (target or src).items()}
    if not target:
        if verbose: print(f"[depth:{stage}] 无市场数据,跳过。")
        return None
    if fixed_sigma is not None: SIGMA = fixed_sigma
    grid = grid or [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10]
    keys = [t for t in target if t in TEAMS]
    if exclude_thesis:
        skip = _exclude_thesis(keys)
        if skip and verbose: print(f"  [depth:{stage}] 剔除(保edge): {sorted(skip)}")
        keys = [t for t in keys if t not in skip]
    best = (None, 1e18)
    for sc in grid:
        ELO_SCALE = sc
        res = simulate(n=n, seed=fit_seed, want_pairs=False)  # CRN
        L = _kl(target, res[stage], keys)
        if verbose: print(f"  [depth:{stage}] ELO_SCALE={sc:4.2f}  KL={L:.4f}")
        if L < best[1]: best = (sc, L)
    if apply: ELO_SCALE = best[0]
    edge = "(边界!扩大grid)" if best[0] in (grid[0], grid[-1]) else ""
    if verbose: print(f"  >>> ELO_SCALE={best[0]:.2f}  ({stage}市场拟合){edge}")
    return best[0]

def calibrate_all(verbose=True):
    """一键:按 单场→出线→深度 的顺序,把有数据的市场各自拟合到对应参数。
       顺序有意义:先定单场粒度(SIGMA),再定小组(GOAL_DIV),最后定整体形状(ELO_SCALE)。"""
    if verbose: print("===== calibrate_all 开始 =====")
    calibrate_match_odds(verbose=verbose)                    # -> SIGMA
    calibrate_group(verbose=verbose)                         # -> GOAL_DIV
    # 深度:优先用更细的层(四强>八强>冠军);谁有数据用谁,冠军兜底
    if REACH_SF_ODDS:      calibrate_depth("sf",   fixed_sigma=SIGMA, verbose=verbose)
    elif REACH_QF_ODDS:    calibrate_depth("qf",   fixed_sigma=SIGMA, verbose=verbose)
    else:                  calibrate_depth("champ",fixed_sigma=SIGMA, verbose=verbose)
    if verbose:
        print(f"===== 完成: SIGMA={SIGMA} GOAL_DIV={GOAL_DIV} "
              f"ELO_SCALE={ELO_SCALE:.2f} =====")

# =====================================================================
#  Brier 校准 SIGMA(历史比赛,数据驱动,与 ELO_SCALE 解耦,用 RAW Elo)
# =====================================================================
def _load_matches_csv(path):
    """CSV 列: elo_a,elo_b,y  或  elo_a,elo_b,score_a,score_b。y=1胜/0.5平/0负。"""
    import csv
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            ea, eb = float(r["elo_a"]), float(r["elo_b"])
            if r.get("y", "") != "": y = float(r["y"])
            else:
                sa, sb = float(r["score_a"]), float(r["score_b"])
                y = 1.0 if sa > sb else (0.0 if sa < sb else 0.5)
            out.append((ea, eb, y))
    return out

def _winprob_vec(diffs, sigma, K, rng_):
    """向量化边际胜率(分块省内存)。"""
    if sigma <= 0: return 1.0 / (1.0 + 10 ** (-diffs / 400))
    n = rng_.normal(0, math.sqrt(2) * sigma, K)
    out = np.empty(len(diffs))
    for i in range(0, len(diffs), 500):
        c = diffs[i:i + 500][:, None]
        out[i:i + 500] = (1.0 / (1.0 + 10 ** (-(c + n[None, :]) / 400))).mean(axis=1)
    return out

def brier_calibrate_sigma(matches, sigma_grid=None, K=8000, seed=0,
                          apply=True, verbose=True):
    """历史比赛用 Brier 反推 SIGMA(用赛前 RAW Elo)。matches: [(elo_a,elo_b,y)] 或 CSV。
       数据来源:比分用 Kaggle 'International football results 1872-2024';
       赛前 Elo 用 eloratings.net 历史评分,按日期 join。优先喂淘汰赛/decisive。"""
    if isinstance(matches, str): matches = _load_matches_csv(matches)
    diffs = np.array([m[0] - m[1] for m in matches], float)
    y = np.array([m[2] for m in matches], float)
    sigma_grid = sigma_grid or [0, 20, 40, 55, 70, 85, 100, 115, 130, 150, 175, 200]
    rng_ = np.random.default_rng(seed); res = []
    for sg in sigma_grid:
        p = _winprob_vec(diffs, sg, K, rng_)
        br = float(np.mean((p - y) ** 2)); res.append((sg, br))
        if verbose: print(f"  SIGMA={sg:4d}  Brier={br:.4f}")
    best = min(res, key=lambda x: x[1])
    if apply:
        global SIGMA; SIGMA = best[0]
    if verbose:
        e = "(边界!扩grid)" if best[0] in (sigma_grid[0], sigma_grid[-1]) else ""
        print(f"  >>> SIGMA={best[0]} Brier={best[1]:.4f} (常数0.5的Brier=0.25){e}")
    return best, res

def _self_test_brier(true_sigma=85, n_matches=6000, seed=1):
    """自检:已知 true_sigma 造合成赛,看能否复原。"""
    r = np.random.default_rng(seed); ms = []
    for _ in range(n_matches):
        ea, eb = r.uniform(1700, 2080), r.uniform(1700, 2080)
        pa = 1.0/(1.0+10**(-((ea+r.normal(0,true_sigma))-(eb+r.normal(0,true_sigma)))/400))
        ms.append((ea, eb, 1.0 if r.random() < pa else 0.0))
    best, _ = brier_calibrate_sigma(ms, apply=False, verbose=False)
    print(f"[自检] true={true_sigma} -> 复原 SIGMA={best[0]} "
          f"{'✓' if abs(best[0]-true_sigma)<=20 else '✗'}")
    return best

# =====================================================================
#  EV / 报告
# =====================================================================
def fair_prob(t, champ):
    m = champ.get(t, 0.0)
    if t in CHAMP_DEVIG:
        return MARKET_BLEND * CHAMP_DEVIG[t] + (1 - MARKET_BLEND) * m
    return m

def report(n=N_SIMS):
    res = simulate(n=n)
    def show(key, title, k=12):
        print(f"\n=== {title} (SCALE={ELO_SCALE:.2f} SIGMA={SIGMA} GOAL_DIV={GOAL_DIV}) ===")
        for t, p in sorted(res[key].items(), key=lambda x: -x[1])[:k]:
            print(f"  {t:13s}{p:6.1%}")
    show("champ", "夺冠概率"); show("final", "进决赛"); show("sf", "四强"); show("qf", "八强")
    print("\n=== 最可能决赛对阵 ===")
    for k, p in sorted(res["pairs"].items(), key=lambda x: -x[1])[:6]:
        print(f"  {k[0]} vs {k[1]:13s}{p:6.1%}")
    print(f"\n=== 冠军盘 EV/Kelly (fair=市场×{MARKET_BLEND:.0%}+模型×{1-MARKET_BLEND:.0%}) ===")
    have = {"Spain":475,"France":500,"England":700,"Argentina":900,"Portugal":1000,
            "Brazil":850,"Germany":1400,"Netherlands":2200,"Norway":3500}  # 示例价,自行更新
    print(f"  {'Team':13s}{'fair':>7s}{'price':>7s}{'EV/$1':>8s}{'¼Kelly':>8s}")
    rows = [(t, fair_prob(t, res["champ"]), a, ev_per_dollar(fair_prob(t,res['champ']),a),
             kelly(fair_prob(t,res['champ']),a)) for t,a in have.items()]
    for t,p,a,ev,kk in sorted(rows, key=lambda x:-x[3]):
        print(f"  {t:13s}{p:6.1%}{('+%d'%a):>7s}{ev:+8.1%}{(f'{kk:5.2%}' if kk>0 else '  —'):>8s}")

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "brier": _self_test_brier(); sys.exit(0)
    if arg == "calibrate": calibrate_all()
    report()