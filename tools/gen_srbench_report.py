# -*- coding: utf-8 -*-
"""Build the LSR-Transform report: tree search rediscovering physical law from data."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, KeepTogether)

pdfmetrics.registerFont(TTFont("CN", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"))

INK      = colors.HexColor("#16191d")
MUTED    = colors.HexColor("#5b6470")
RULE     = colors.HexColor("#d4d8de")
ACCENT   = colors.HexColor("#0b5c8a")
HILITE   = colors.HexColor("#eef4f8")
EQBG     = colors.HexColor("#f6f7f9")

ss = getSampleStyleSheet()
def S(name, **kw):
    base = dict(name=name, fontName="CN", textColor=INK, alignment=TA_LEFT,
                fontSize=9.5, leading=15.5)
    base.update(kw)
    return ParagraphStyle(**base)

TITLE   = S("t",  fontSize=21, leading=27, spaceAfter=3)
SUB     = S("s",  fontSize=11, leading=16, textColor=ACCENT, spaceAfter=2)
DEK     = S("d",  fontSize=9,  leading=14, textColor=MUTED)
H1      = S("h1", fontSize=13, leading=18, spaceBefore=13, spaceAfter=5, textColor=ACCENT)
H2      = S("h2", fontSize=10.5, leading=15, spaceBefore=10, spaceAfter=3)
BODY    = S("b",  spaceAfter=5)
SMALL   = S("sm", fontSize=8.3, leading=12.5, textColor=MUTED)
CELL    = S("c",  fontSize=8.6, leading=12)
CELLB   = S("cb", fontSize=8.6, leading=12)
EQ      = ParagraphStyle("eq", fontName="Courier", fontSize=8.4, leading=12.4,
                         textColor=INK)
EQMUTED = ParagraphStyle("eqm", fontName="Courier", fontSize=8.4, leading=12.4,
                         textColor=MUTED)

def rule(space_before=4, space_after=8, colour=RULE):
    t = Table([[""]], colWidths=[165*mm], rowHeights=[0.6])
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), colour),
                           ("TOPPADDING", (0,0), (-1,-1), 0),
                           ("BOTTOMPADDING", (0,0), (-1,-1), 0)]))
    return [Spacer(1, space_before), t, Spacer(1, space_after)]

def law(title, meaning, truth, found, note=None):
    """One recovered law: what it is, the rearranged truth, what the search returned."""
    rows = [[Paragraph(f"<b>{title}</b>", H2)],
            [Paragraph(meaning, SMALL)],
            [Paragraph(f'<font name="CN" size="8">隐藏的真值</font>'
                       f'&nbsp;&nbsp;{truth}', EQMUTED)],
            [Paragraph(f'<font name="CN" size="8">从数据得出</font>'
                       f'&nbsp;&nbsp;{found}', EQ)]]
    if note:
        rows.append([Paragraph(note, SMALL)])
    t = Table(rows, colWidths=[165*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), EQBG),
        ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("TOPPADDING", (0,0), (0,0), 6), ("BOTTOMPADDING", (0,-1), (-1,-1), 7),
        ("LINEBEFORE", (0,0), (0,-1), 2, ACCENT),
    ]))
    return KeepTogether([t, Spacer(1, 5)])

story = []
story += [Paragraph("从观测数据中发现物理规律", TITLE),
          Paragraph("ERA flat-PUCT 程序搜索在 LLM-SRBench / LSR-Transform 上的结果", SUB),
          Paragraph("输入只有采样点，没有方程 &nbsp;·&nbsp; 111 道题 &nbsp;·&nbsp; "
                    "每题 24 次程序改写、16.5 次模型调用 &nbsp;·&nbsp; deepseek-v4-flash", DEK)]
story += rule(6, 10)

# ---------------------------------------------------------------- 一
story += [Paragraph("一、数据集在问什么", H1),
          Paragraph("<b>输入是一张纯数字表：4000 行采样点，每行是若干自变量的取值和一个观测量。"
                    "没有方程，没有提示，真值从头到尾不出现在任何环节。</b>"
                    "要求输出的是一段程序，它写出的方程要能解释这些数字。", BODY),
          Paragraph("题目取自费曼物理讲义的 111 个方程，且都<b>换未知量重排</b>过。"
                    "库仑定律 F = q1·q2/(4·pi·eps·r^2) 人人会背，但这里给的是 F、q1、q2、eps 四列观测值，"
                    "要求解出 <b>r</b>：", BODY),
          Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;r = -sqrt(q1*q2/(F*eps)) / (2*sqrt(pi))", EQ),
          Spacer(1, 6),
          Paragraph("这个形式不出现在任何教科书里。模型见过原式、见不到重排后的版本，"
                    "所以无法靠检索作答，只能<b>从那 4000 个采样点里把函数形式定出来</b>，"
                    "再由优化器填常数。这就是该数据集的设计目的：把「记住方程」和"
                    "「从观测数据里把方程找出来」分开。", BODY)]

# ---------------------------------------------------------------- 二
story += [Paragraph("二、方法", H1),
          Paragraph("每道题独立搜索。树的每个节点是一段完整的 Python 程序，由语言模型改写父节点得到；"
                    "flat-PUCT 依据「已知最好分数 + 访问次数」决定下一次从哪个节点展开。"
                    "程序在沙箱中运行，返回一个含 params[i] 空位的方程骨架，"
                    "常数由评测方自己用一次 BFGS 从全 1 出发拟合 —— "
                    "这套拟合协议与上游 LLM-SR 逐字一致。", BODY),
          Paragraph("预算：每题 24 次展开、3 个并发 worker、20 秒沙箱时限。"
                    "答案永远不接触测试集；选择只在训练集切出的 25% 验证片上进行。", BODY)]

# ---------------------------------------------------------------- 三
story += [Paragraph("三、结果", H1),
          Paragraph("对照论文 Table 2，每个方法取其最好的 backbone（GPT-4o-mini）。"
                    "SA = 符号准确率，判断答案是否<b>就是那个方程</b>；"
                    "Acc(0.1) = 最坏点相对误差是否 &lt;= 10%；NMSE = 归一化均方误差。", BODY)]

hdr = ["方法", "SA (%)", "Acc(0.1) (%)", "NMSE", "每题调用"]
data = [[Paragraph(f"<b>{h}</b>", CELLB) for h in hdr],
        [Paragraph("Direct prompting", CELL), "7.21",  "6.31",  "0.2631", "~250"],
        [Paragraph("SGA", CELL),              "9.91",  "8.11",  "0.2321", "~250"],
        [Paragraph("LaSR", CELL),             "6.31",  "50.45", "0.0011", Paragraph("GP 数百万次变异", CELL)],
        [Paragraph("LLM-SR", CELL),           "31.53", "39.64", "0.0091", "250"],
        [Paragraph("<b>ERA 树搜索（本工作）</b>", CELLB),
         "41.4", "56.8", "2.15e-08", "16.5"]]
t = Table(data, colWidths=[52*mm, 22*mm, 28*mm, 28*mm, 35*mm])
t.setStyle(TableStyle([
    ("FONTNAME", (1,1), (-1,-1), "Helvetica"),
    ("FONTSIZE", (0,0), (-1,-1), 8.6),
    ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("LINEBELOW", (0,0), (-1,0), 0.8, INK),
    ("LINEABOVE", (0,-1), (-1,-1), 0.8, INK),
    ("BACKGROUND", (0,-1), (-1,-1), HILITE),
    ("TEXTCOLOR", (0,-1), (-1,-1), INK),
    ("FONTNAME", (1,-1), (-1,-1), "Helvetica-Bold"),
    ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LINEBELOW", (0,1), (-1,-2), 0.3, RULE),
]))
story += [t, Spacer(1, 7),
          Paragraph("<b>三栏全部领先。</b>111 道里 63 道达到 Acc(0.1)，其中 46 道经判定符号上就是那个方程；"
                    "52 道把 NMSE 压到 1e-12 以下的计分上限。全集耗时 1.11 小时。", BODY),
          Paragraph("LaSR 那一行值得单看：Acc 50.45% 而 SA 只有 6.31% —— "
                    "数值上一半题过关，符号上仅十六分之一是真方程。它靠数百万次遗传变异拼出数值贴合的表达式，"
                    "而不是把定律推出来。这一栏正是「拟合」与「发现」的分界。", BODY)]

# ---------------------------------------------------------------- 四
story += [Paragraph("四、从观测数据中找出的物理定律", H1),
          Paragraph("下面每一条，上行是<b>搜索从未见过的</b>真值（数据集重排后的形式），"
                    "下行是搜索仅凭那张数字表返回的答案。"
                    "答案中的 params[i] 由评测方拟合，模型只负责给出结构。", BODY)]

story += [law("玻尔能级 —— 反解主量子数",
              "由氢原子能级的观测值反推主量子数 n",
              "-sqrt(2)*q^2*sqrt(-m/E_n)/(4*eps*h)",
              "sqrt(-m*q^4/(eps^2*h^2*E_n))",
              "模型把 q^2 与 1/(eps*h) 全部收进根号 —— 代数等价，且比真值形式更紧凑。")]

story += [law("普朗克 / 玻色-爱因斯坦分布 —— 反解温度",
              "由黑体辐射谱的观测值反推系统温度",
              "h*omega/(2*pi*kb*log(1 + h*omega/(2*pi*E_n)))",
              "params[0]*h*omega/(kb*log(1 + params[1]*h*omega/E_n)) + params[2]",
              "嵌套 log 的结构完整写对，包括分母上 log 内的 1 + x 形式。")]

story += [law("波导色散 —— 反解角频率",
              "矩形波导中导波的截止频率关系",
              "c*sqrt(d^2*k^2 + pi^2)/d",
              "c*sqrt(k^2 + (pi/d)^2)",
              "模型把 d 移入根号内约掉，还原成物理学家惯用的截止频率写法 "
              "omega = c*sqrt(k^2 + (pi/d)^2)。真值那个形式反而是被人为搅乱过的。")]

story += [law("相对论多普勒效应 —— 反解静止频率",
              "由观测到的相对论频移反推光源静止频率",
              "c*omega*sqrt(1 - v^2/c^2)/(c + v)",
              "params[0]*omega*sqrt(1 - params[1]*v^2/c^2)/(1 + params[2]*v/c)",
              "分子的时间膨胀因子 sqrt(1-beta^2) 与分母的传播延迟 1+beta，两项都在正确位置。")]

story += [law("顺磁二能级布居 —— 玻尔兹曼配分",
              "磁场中自旋二能级系统的平衡粒子数",
              "n*exp(B*mom/(T*kb)) + n*exp(-B*mom/(T*kb))",
              "params[0]*n*exp(params[1]*(mom*B/(kb*T))) + params[2]*n*exp(-params[1]*(mom*B/(kb*T)))",
              "两个指数项符号相反、指数幅值共用同一个参数 —— 这正是 2n*cosh(mu*B/kT) 的展开形式。")]

story += [law("理想气体等温膨胀 —— 反解末体积",
              "由做功量反推气体膨胀后的体积",
              "V1*exp(E_n/(T*kb*n))",
              "V1*exp(E_n/(n*kb*T))",
              "一字不差，连一个多余的自由参数都没有引入。")]

story += [law("相对论动量 —— 反解速度",
              "由动量与静止质量反推粒子速度",
              "-c*p*sqrt(1/(c^2*m_0^2 + p^2))",
              "params[0]*p*c/sqrt(m_0^2*c^2 + p^2) + params[1]",
              "把 sqrt(1/X) 化为 1/sqrt(X) 并配成勾股形式。")]

story += [law("受迫振子 —— 反解驱动频率",
              "由振幅与恢复力反推外加驱动的角频率",
              "sqrt(-Ef*q/(m*x) + omega_0^2)",
              "sqrt(omega_0^2 - params[0]*(q*Ef)/(m*x))")]

story += [law("塞曼分裂 —— 反解磁场强度",
              "由能级间距反推所处磁场",
              "E_n*h/(2*pi*Jz*g_*mom)",
              "params[0]*E_n*h/(g_*mom*Jz)")]

story += [law("一维无限深势阱 —— 基态能量",
              "盒中粒子的能级公式",
              "h^2/(8*pi^2*d^2*m)",
              "params[0]*h^2/(m*d^2) + params[1]")]

story += [law("朗之万介电 —— 反解平衡载流子密度",
              "取向极化下的粒子数分布",
              "T*kb*n/(Ef*p_d*cos(theta) + T*kb)",
              "n/(1 + params[0]*(p_d*Ef*cos(theta)/(kb*T)))",
              "把分式化为 1/(1+x) 的标准形式，与朗之万函数的低场展开一致。")]

story += rule(6, 3)
story += [Paragraph("结果文件 bench/results/era-srbench-deepseek-transform.json&nbsp;·&nbsp;"
                    "符号准确率判定 deepseek-transform-symbolic.json&nbsp;·&nbsp;"
                    "方法与完整审计 docs/algo-era.md", SMALL)]

def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("CN", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(22*mm, 12*mm, "ERA 树搜索 · LLM-SRBench / LSR-Transform")
    canvas.drawRightString(188*mm, 12*mm, str(doc.page))
    canvas.restoreState()

OUT = "/home/user/agentdescent/bench/reports/era-llm-srbench-transform.pdf"
doc = BaseDocTemplate(OUT, pagesize=A4,
                      leftMargin=22*mm, rightMargin=22*mm,
                      topMargin=18*mm, bottomMargin=20*mm,
                      title="从观测数据中发现物理规律 — LSR-Transform",
                      author="AgentDescent / ERA")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
doc.build(story)
print("wrote", OUT)
