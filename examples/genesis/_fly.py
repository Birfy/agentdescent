"""第五个域：一份需求，一个冻结的驱动，其余全部生长 —— 包括测试。

前四个域生长一个**库**，并且人写好整套测试再把测试的**源码**贴进提示词。上游不是这样做
的：那里的 agent 自己写测试，自己跑 `mix test`，"tests are the definition of done"
(`agents/manager.ex`)；Genesis 验证用的 c-testsuite、LLVM、Csmith 是实验者**事后**拿来量
的外部基准，从不进 agent 的内循环。把断言的源码交给 agent，它就不再是在实现需求，而是在对
着断言写代码。

所以这个域只给两样东西，两样都是**需求**而不是设计：

* `REQUIREMENTS.md` —— 用户要什么，用户的原话。要一只连接组级真实的果蝇大脑、三种任务、
  学习要能改变行为、要前后端、每个目录要为自己的文件写测试。外加一份文献清单。里面没有方
  程、没有系数、没有模块划分。
* `fly.py` —— 那份需求的可执行形式，冻结。它只 import `src` 的**四个**名字
  (`make_brain`、`train`、`run_episode`、`handle`)，所以公开接口也只有这四个；分几层、谁
  调用谁、写哪些测试，全是它们的内部事务。

其余一切生长：`CONTEXT.md` 树由 `--mode b` 的 architect 设计，电路、模块、前端、以及**仓
库里的每一个测试**，都是 agent 写的。

两套断言都在仓库外面。`hidden` 是黑盒验收，驱动搜索 —— agent 看到的是需求标题、断言的名
字和它报了什么，看不到一行源码。`audit` 是**封存**的事后测量：同一批要求换了问法、换了种
子、换了任务，开跑前写死，跑完只用来报一个数。"验收全过"和"封存套件全过"之间的差距，就是
它把东西做对了、还是只对着验收写的度量。

有一条要求这两套都测不了，写在这里不藏着：**大脑是不是真的照连接组建的**。黑盒问不出来。
能问的只有父节点评审读 diff（`ParentCodeReview`）和人读结果。
"""

from __future__ import annotations

from typing import Dict, List, Mapping

from ._suite import BLIND_FAILURE, PYTHON_MODULE_SKILL, TestSuite, reward_test
from ._suite import llm_executor as _llm_executor
from ._suite import llm_manager as _llm_manager
from ._world import SKILLS_DIR

__all__ = ["CASE_NOUN", "CONTRACTS", "FLY", "FROZEN", "GROUP_NOUN", "HELD_OUT_FRAC",
           "OBJECTIVE", "REQUIRES_MODEL", "SCORING", "build_tasks", "initial_files",
           "FAILURE", "llm_executor", "llm_manager", "make_runner", "own_review",
           "reward",
           "suite_failures", "suite_review"]

#: 人写的全部，一个文件。拒绝给任何提案，评分前恢复原样。
#:
#: `fly.py` 曾经也在这里，而它不该在。这个 port 自己的偏差表就是这么记的 ——
#: "a frozen entry point, so the result is a program rather than a package nobody can
#: invoke. **A choice, and a defensible one**" —— 一个选择，不是上游有的东西。上游
#: Mode B 的输入只有一句 objective，c-testsuite / LLVM / Csmith 是实验者事后拿来量的
#: 外部基准，不在仓库里。
#:
#: 上游其实也有契约，只是隐含在目标里："造一个 C 编译器"这句话本身就钉死了 `cc -o foo
#: foo.c`，一个人人都知道的约定，所以 c-testsuite 能跑起来。而"一个果蝇模拟器"没有任何
#: 公认的调用方式，所以那个约定得说出来 —— 但它属于**需求**（我要怎么用它），不属于
#: 仓库（这是给你们的代码）。现在它在 `REQUIREMENTS.md` 的第四节里，`fly.py` 由它们写。
FROZEN = ("REQUIREMENTS.md",)

#: 推进每一个 brief 的东西。
CONTRACTS = ("REQUIREMENTS.md",)

SCORING = "黑盒验收（agent 读不到），外加一套开跑前封存的事后测量"
CASE_NOUN = "验收断言"
GROUP_NOUN = "验收文件"

ENTRY = "src/__init__.py"

#: 没有参考实现，也就没有离线规则化 actor；写一个出来就等于把该生长的设计先写了。
REQUIRES_MODEL = True

OBJECTIVE = (
    "实现 `REQUIREMENTS.md` 要的那个软件：把一只果蝇的大脑照真实连接组建出来，让它在竞技"
    "场里执行任务、领取奖惩、并因此改变行为，再把整个东西作为一个网页端出来。库放在 "
    "`src/` 下面。仓库根目录"
    "下要有一个 `fly.py`，能跑 `train`、`probe`、`serve` 三个子命令，三个都认 "
    "`--format json`；那个文件也是你们写的。除此之外 —— 分几层、每层叫什么、谁调用谁、"
    "公开接口长什么样 —— 全部由你们决定。可以用 numpy 和其它第三方包。"
    "**每个目录都要为自己的文件写测试**：测试是“做完了”的定义，一个写了实现却没有测试的"
    "节点，它的上级没有任何东西可以运行来验收它。"
)

#: 每个验收文件是从需求的哪一条来的 —— blind 提示词由它加上断言名拼成。
REQUIREMENTS = {
    "training": "需求 二、四：三种任务都要能训练；`python fly.py train` 要能跑，"
                "每一局报出它挨了多少惩罚、拿了多少回报、走了多少步。同一个种子给同一条轨迹。",
    "learning": "需求 二：学习必须改变行为，而且要能被看见。同一个气味，训练前后动物的"
                "反应应当不同；关掉可塑性时动物不得被改变。",
    "page": "需求 三：一个网页，打开就能看 —— 动物在竞技场里走、大脑各层此刻的活动、"
            "学习曲线。",
    "api": "需求 三：后端要能取当前状态、单步推进、重置、跑训练；非法路由和非法请求体"
           "要被拒绝。",
    "sealed": "需求 二、三：同样的要求，换一种问法。",
}

#: 黑盒验收，驱动搜索。**不在仓库里** —— 只在一次评分的临时副本里出现。
_ACCEPTANCE = {
    'acceptance/_cli.py': r'''"""跑那个程序，看它说了什么。没有一行 import 它们写的模块。

上游的验证就是这个形状 —— 跑二进制、检查输出（c-testsuite / LLVM / Csmith）。这里唯一的
契约是 `REQUIREMENTS.md` 里那三条命令行，`src/` 底下怎么分层完全不关这些断言的事。
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

TASKS = ("taxis", "avoid", "choice")
TIMEOUT = 300


def run(*args, timeout=TIMEOUT):
    """跑 `python fly.py ...`，返回 (returncode, stdout, stderr)。"""
    done = subprocess.run([sys.executable, "fly.py", *args],
                          capture_output=True, text=True, timeout=timeout)
    return done.returncode, done.stdout, done.stderr


def json_run(*args, timeout=TIMEOUT):
    """同上，但要求 `--format json` 的输出是一个能解析的 JSON，且 stdout 只有它。"""
    code, out, err = run(*args, "--format", "json", timeout=timeout)
    assert code == 0, f"fly.py {' '.join(args)} 退出 {code}\n{err[-600:]}"
    return json.loads(out.strip())


def serving(port, *, timeout=40):
    """起 `serve`，等它能应答，交出一个取 URL 的函数。"""
    proc = subprocess.Popen([sys.executable, "fly.py", "serve", "--port", str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"serve 退出了 {proc.returncode}: "
                                 f"{(proc.stderr.read() or '')[-600:]}")
        try:
            urllib.request.urlopen(base + "/", timeout=2).read()
            return proc, base
        except urllib.error.HTTPError:
            return proc, base
        except Exception:
            time.sleep(0.4)
    proc.kill()
    raise AssertionError(f"serve 在 {timeout}s 内没有应答")


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, r.headers.get("content-type", ""), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("content-type", ""), e.read().decode()


def post(base, path, payload):
    req = urllib.request.Request(base + path, method="POST",
                                 data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
''',
    'acceptance/test_api.py': r'''"""需求 三：后端要能取状态、单步、重置、训练。"""

import json

from _cli import get, post, serving


def test_the_api_reports_where_the_animal_is_and_what_its_brain_is_doing():
    proc, base = serving(8733)
    try:
        status, content_type, body = get(base, "/api/state")
        assert status == 200 and "json" in content_type
        state = json.loads(body)
        assert "position" in state and len(state["position"]) == 2
        assert isinstance(state.get("brain"), dict) and len(state["brain"]) >= 3
    finally:
        proc.kill()


def test_the_api_can_step_and_reset():
    proc, base = serving(8734)
    try:
        post(base, "/api/reset", {"task": "avoid", "seed": 0})
        before = json.loads(get(base, "/api/state")[2])["step"]
        post(base, "/api/step", {"steps": 10})
        after = json.loads(get(base, "/api/state")[2])["step"]
        assert after > before
        post(base, "/api/reset", {"task": "avoid", "seed": 0})
        assert json.loads(get(base, "/api/state")[2])["step"] == 0
    finally:
        proc.kill()


def test_the_api_can_train():
    proc, base = serving(8735)
    try:
        status, _ = post(base, "/api/train", {"task": "avoid", "episodes": 3})
        assert status == 200
    finally:
        proc.kill()


def test_an_unknown_route_is_refused():
    proc, base = serving(8736)
    try:
        assert get(base, "/api/nothing-here")[0] == 404
    finally:
        proc.kill()
''',
    'acceptance/test_learning.py': r'''"""需求 二：学习必须改变行为，而且要能被看见。"""

from _cli import json_run


def test_a_fresh_animal_is_punished_in_the_avoidance_assay():
    """没学过的动物会一头撞进惩罚区 —— 否则这个实验什么也测不到。"""
    result = json_run("probe", "--task", "avoid", "--episodes", "1", "--trials", "3")
    assert result["naive"] > 0


def test_training_reduces_the_punishment_the_animal_takes():
    rows = json_run("train", "--task", "avoid", "--episodes", "12")
    shocks = [r["shocks"] for r in rows]
    assert sum(shocks[:3]) > 0
    assert sum(shocks[-3:]) < sum(shocks[:3])


def test_what_was_learned_transfers_to_arenas_it_never_saw():
    """关掉可塑性、用没训练过的种子 —— 唯一诚实的学习度量。"""
    result = json_run("probe", "--task", "avoid", "--episodes", "10", "--trials", "5")
    assert result["naive"] > 0
    assert result["trained"] < result["naive"]


def test_learning_shows_up_in_the_choice_assay_as_well():
    result = json_run("probe", "--task", "choice", "--episodes", "10", "--trials", "5")
    assert result["trained"] <= result["naive"]
''',
    'acceptance/test_page.py': r'''"""需求 三：一个网页，打开就能看。"""

from _cli import get, serving


def test_the_page_is_served_at_the_root():
    proc, base = serving(8731)
    try:
        status, content_type, body = get(base, "/")
        assert status == 200
        assert "html" in content_type.lower()
        assert body.lstrip().lower().startswith("<!doctype html")
    finally:
        proc.kill()


def test_the_page_draws_the_arena_the_brain_and_the_learning_curve():
    proc, base = serving(8732)
    try:
        _, _, page = get(base, "/")
        low = page.lower()
        assert "canvas" in low
        for word in ("arena", "brain", "learn"):
            assert word in low, word
    finally:
        proc.kill()
''',
    'acceptance/test_program.py': r'''"""需求 四：我要能跑它。"""

from _cli import TASKS, json_run, run


def test_the_program_exists_and_explains_itself():
    code, out, err = run("--help")
    assert code == 0, err[-400:]
    for word in ("train", "probe", "serve"):
        assert word in (out + err)


def test_every_task_can_be_trained():
    for task in TASKS:
        rows = json_run("train", "--task", task, "--episodes", "2")
        assert isinstance(rows, list) and len(rows) == 2, task


def test_each_episode_reports_its_shocks_reward_and_length():
    for i, row in enumerate(json_run("train", "--task", "avoid", "--episodes", "3")):
        assert row["episode"] == i
        assert row["shocks"] >= 0
        assert isinstance(row["reward"], (int, float))
        assert row["steps"] > 0


def test_the_same_seed_gives_the_same_run():
    one = json_run("train", "--task", "avoid", "--episodes", "4", "--seed", "0")
    two = json_run("train", "--task", "avoid", "--episodes", "4", "--seed", "0")
    assert [r["shocks"] for r in one] == [r["shocks"] for r in two]


def test_json_output_is_only_json():
    """我要拿它画曲线，所以 stdout 上不能混别的东西。"""
    code, out, err = run("train", "--task", "taxis", "--episodes", "1", "--format", "json")
    assert code == 0
    assert out.strip().startswith(("[", "{"))
''',
}

#: 封存的事后测量。开跑前写死，全程不进仓库，跑完只用来报一个数。
_SEALED = {
    'sealed/_cli.py': r'''"""跑那个程序，看它说了什么。没有一行 import 它们写的模块。

上游的验证就是这个形状 —— 跑二进制、检查输出（c-testsuite / LLVM / Csmith）。这里唯一的
契约是 `REQUIREMENTS.md` 里那三条命令行，`src/` 底下怎么分层完全不关这些断言的事。
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

TASKS = ("taxis", "avoid", "choice")
TIMEOUT = 300


def run(*args, timeout=TIMEOUT):
    """跑 `python fly.py ...`，返回 (returncode, stdout, stderr)。"""
    done = subprocess.run([sys.executable, "fly.py", *args],
                          capture_output=True, text=True, timeout=timeout)
    return done.returncode, done.stdout, done.stderr


def json_run(*args, timeout=TIMEOUT):
    """同上，但要求 `--format json` 的输出是一个能解析的 JSON，且 stdout 只有它。"""
    code, out, err = run(*args, "--format", "json", timeout=timeout)
    assert code == 0, f"fly.py {' '.join(args)} 退出 {code}\n{err[-600:]}"
    return json.loads(out.strip())


def serving(port, *, timeout=40):
    """起 `serve`，等它能应答，交出一个取 URL 的函数。"""
    proc = subprocess.Popen([sys.executable, "fly.py", "serve", "--port", str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"serve 退出了 {proc.returncode}: "
                                 f"{(proc.stderr.read() or '')[-600:]}")
        try:
            urllib.request.urlopen(base + "/", timeout=2).read()
            return proc, base
        except urllib.error.HTTPError:
            return proc, base
        except Exception:
            time.sleep(0.4)
    proc.kill()
    raise AssertionError(f"serve 在 {timeout}s 内没有应答")


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, r.headers.get("content-type", ""), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("content-type", ""), e.read().decode()


def post(base, path, payload):
    req = urllib.request.Request(base + path, method="POST",
                                 data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
''',
    'sealed/test_sealed.py': r'''"""封存的事后测量：开跑前写死，全程不进仓库，任何 agent 读不到，跑完只用来报一个数。

上游就是这个协议 —— c-testsuite / LLVM / Csmith 是实验者事后拿来量的外部基准，从不进
agent 的内循环。这套问的是和验收同一批要求，但换了种子、换了任务、换了问法。"验收全过"
和"这套全过"之间的差距，就是它把东西做出来了、还是只对着验收写的度量。

有一条这套测不了，写在这里不藏着：**大脑是不是真的照连接组建的**。跑二进制问不出来，只能
靠父节点读 diff 和人读结果。
"""

import json

from _cli import get, json_run, post, run, serving


def test_the_gain_from_training_survives_a_different_seed():
    result = json_run("probe", "--task", "avoid", "--episodes", "10",
                      "--trials", "6", "--seed", "5")
    assert result["naive"] > 0
    assert result["trained"] < 0.6 * result["naive"]


def test_more_training_is_not_worse_than_less():
    short = json_run("probe", "--task", "avoid", "--episodes", "4", "--trials", "5", "--seed", "2")
    long = json_run("probe", "--task", "avoid", "--episodes", "16", "--trials", "5", "--seed", "2")
    assert long["trained"] <= short["trained"]


def test_the_taxis_assay_is_actually_reachable():
    """趋向实验要真的能到达 —— 每局都跑满上限说明动物根本没找到源。"""
    rows = json_run("train", "--task", "taxis", "--episodes", "6")
    steps = [r["steps"] for r in rows]
    assert min(steps) < max(steps) or min(steps) < 400


def test_a_different_seed_is_a_different_animal():
    one = json_run("train", "--task", "avoid", "--episodes", "4", "--seed", "11")
    two = json_run("train", "--task", "avoid", "--episodes", "4", "--seed", "22")
    assert [r["shocks"] for r in one] != [r["shocks"] for r in two]


def test_training_on_one_task_does_not_break_another():
    """不应该为了通过一个实验把另一个弄坏。"""
    rows = json_run("train", "--task", "choice", "--episodes", "3", "--seed", "7")
    assert all(r["steps"] > 0 for r in rows)


def test_every_task_survives_a_longer_run():
    for task in ("taxis", "avoid", "choice"):
        rows = json_run("train", "--task", task, "--episodes", "8", "--seed", "3")
        assert len(rows) == 8, task


def test_a_bad_task_name_is_refused_rather_than_pretended():
    code, out, err = run("train", "--task", "fly-to-the-moon", "--episodes", "1")
    assert code != 0


def test_the_state_the_page_draws_changes_as_the_animal_moves():
    proc, base = serving(8741)
    try:
        post(base, "/api/reset", {"task": "taxis", "seed": 0})
        first = json.loads(get(base, "/api/state")[2])
        post(base, "/api/step", {"steps": 25})
        later = json.loads(get(base, "/api/state")[2])
        assert first["position"] != later["position"]
    finally:
        proc.kill()


def test_the_page_exposes_more_than_one_layer_of_the_brain():
    """需求里写了"要能看见大脑各层此刻的活动"，所以状态里不能只有一个标量。"""
    proc, base = serving(8742)
    try:
        post(base, "/api/step", {"steps": 5})
        brain = json.loads(get(base, "/api/state")[2])["brain"]
        populations = [k for k, v in brain.items() if isinstance(v, (list, dict)) and v]
        assert len(populations) >= 3, sorted(brain)
    finally:
        proc.kill()


def test_the_server_says_where_it_is_listening():
    """`serve --format json` 要打出一行 URL，否则脚本没法自动接上它。"""
    import subprocess, sys, time
    proc = subprocess.Popen([sys.executable, "fly.py", "serve", "--port", "8743",
                             "--format", "json"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 30
        line = ""
        while time.time() < deadline and not line.strip():
            line = proc.stdout.readline()
            if proc.poll() is not None:
                break
        assert "8743" in line and json.loads(line.strip()).get("url")
    finally:
        proc.kill()
''',
}

#: 用户的要求，原话。人写的唯一一个文件。
_REQUIREMENTS = r'''# 需求：一个果蝇大脑的模拟软件

我要一个软件，把一只果蝇（*Drosophila melanogaster*）建模出来：它的大脑、它的身体、
它要做的事，以及它怎么从做事的结果里学到东西。我要能在浏览器里看着它学。

这份文件是**需求**，不是设计。怎么分层、怎么划模块、写哪些测试、页面怎么排，都由你们决定。

## 一、大脑要复杂，而且要真实

这是最重要的一条。我不要一个"能跑通任务的神经网络"，我要一个**照着真实果蝇大脑建的**模型：

- 神经元的分群、数量、连接极性（兴奋还是抑制）、脑区划分，要能对得上已发表的连接组和电路
  生理工作，不能是拍脑袋编的。
- 代码里要出现**真实的细胞类型名**，并注明它在真实果蝇里有多少个；模型里如果缩小了规模，
  要写清楚真实数字是多少、为什么缩小。
- 涉及嗅觉学习和导航的主要脑区都要有，不能只做一条通路。
- 参考下面那份文献清单，以最新的为准。哪一条设计是从哪篇来的，写在对应的 `CONTEXT.md` 里。

**做不到的地方要说出来**，写进 `## Known Issues`，不要用一个看起来像那么回事的近似糊过去
然后当成真的。

## 二、它要能执行任务、拿到反馈、并因此改变行为

至少三种任务，覆盖三种不同的情形：

1. 趋向一个气味源；
2. 学会**回避**一个会带来惩罚的气味源；
3. 在两个气味源之间做选择，其中一个有惩罚。

关键要求：**学习必须改变行为，而且要能被看见**。同一个气味，训练前后动物的反应应当不同；
没有被训练过的气味不应该受影响。

奖惩按时间步给，不是按整局给。

## 三、前端和后端

**后端**：一套 HTTP 接口，能取当前状态、单步推进、重置、跑训练。

**前端**：一个网页，打开就能看。要能看见的东西：动物在竞技场里走；大脑各层此刻的活动；
学习曲线。控件至少要能选任务、播放/暂停、单步、训练、换一只没学过的新大脑。

手机宽度下要能用，跟随系统的明暗配色。不要依赖 CDN，不要在页面里访问外网。

## 四、我要怎么用它

我要的是一个**能跑的程序**，不是一个没人能调用的包。仓库根目录下要有一个 `fly.py`，
我用它就能做完下面三件事 —— 这个文件也是你们写的，我只说我要怎么用：

```
python fly.py train --task avoid --episodes 20
python fly.py probe --task avoid --episodes 10 --trials 5
python fly.py serve --port 8000
```

**train** 跑若干局并把每一局打出来：第几局、挨了多少次惩罚、拿了多少回报、走了多少步。

**probe** 训练一只，再把它和一只没学过的放进同样的、没训练过的竞技场，**关掉可塑性**，
各跑几局，报出两边各挨了多少惩罚。这是唯一诚实的学习度量：同样的环境、没见过的种子、
测试期间什么都不改变。

**serve** 起一个 HTTP 服务，浏览器打开就能看。

三个子命令都要认 `--format json`：这时候**只往 stdout 打一个 JSON**，别的什么都不打。
我要拿它画学习曲线、做对比，所以得是机器能读的：

- `train --format json` → 一个数组，每局一个对象，至少含 `episode`（从 0 数）、
  `shocks`、`reward`、`steps`
- `probe --format json` → 一个对象，至少含 `naive` 和 `trained`，各是那一边挨的总惩罚数
- `serve --format json` → 打出一行 `{"url": "http://..."}` 然后继续服务

任务名至少要有 `taxis`、`avoid`、`choice` 三个。

## 五、工程要求

- Python 3。**可以用 numpy 和其它第三方包**；用了什么，写在 `requirements.txt` 里。
- **确定性**：同一个种子，任何机器上给出同一条轨迹。
- **每个目录都要为自己的文件写测试。** 测试是"做完了"的定义 —— 一个写了实现却没有测试的
  节点，它的上级没有任何东西可以运行来验收它。测试放在它覆盖的代码旁边。
- 不要留半个文件。不要留没人 import 的死模块。

## 参考文献

连接组与全脑模型
- Dorkenwald 等，*Neuronal wiring diagram of an adult brain*，Nature 2024（FlyWire）
- Schlegel 等，*Whole-brain annotation and multi-connectome cell typing of Drosophila*，Nature 2024
- Shiu 等，*A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain*，Nature 2024
- Lappalainen 等，*Connectome-constrained networks predict neural activity across the fly visual system*，Nature 2024

嗅觉、蘑菇体与学习
- Olsen、Bhandawat & Wilson，*Divisive normalization in olfactory population codes*，Neuron 2010
- Caron 等，*Random convergence of olfactory inputs in the Drosophila mushroom body*，Nature 2013
- Zheng 等，*Structured sampling of olfactory input by the fly mushroom body*，Current Biology 2022
- Aso 等，*The neuronal architecture of the mushroom body provides a logic for associative learning*，eLife 2014
- Hige 等，*Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila*，Neuron 2015
- Handler 等，*Distinct dopamine receptor pathways underlie the temporal sensitivity of associative learning*，Cell 2019
- Felsenberg 等，*Integration of parallel opposing memories underlies memory extinction*，Cell 2018
- Dolan 等，*Neurogenetic dissection of the Drosophila lateral horn*，eLife 2019

中央复合体与导航
- Seelig & Jayaraman，*Neural dynamics for landmark orientation and angular path integration*，Nature 2015
- Green 等，*A neural circuit architecture for angular integration in Drosophila*，Nature 2017
- Turner-Evans 等，*The neuroanatomical ultrastructure and function of a biological ring attractor*，Neuron 2020
- Hulse 等，*A connectome of the Drosophila central complex*，eLife 2021
- Lyu、Abbott & Maimon，*Building an allocentric travelling direction signal via vector computation*，Nature 2022
- Lu 等，*Transforming representations of movement from body- to world-centric space*，Nature 2022
- Westeinde 等，*Transforming a head direction signal into a goal-oriented steering command*，Nature 2024
'''

def initial_files() -> Dict[str, str]:
    """整个仓库的起点：一份需求，和一份写模块的须知。没有驱动，没有测试，没有记录。"""
    return {"REQUIREMENTS.md": _REQUIREMENTS,
            f"src/{SKILLS_DIR}/python-modules.md": PYTHON_MODULE_SKILL}


FLY = TestSuite(name="fly", given=initial_files(), frozen=FROZEN,
                hidden=_ACCEPTANCE, audit=_SEALED, requirements=REQUIREMENTS)

build_tasks = FLY.build_tasks
make_runner = FLY.make_runner
suite_review = FLY.review
#: 父节点评审的另一半：跑**子节点自己写的**测试，并打回写了实现却没写测试的节点。
own_review = FLY.own_review
def suite_failures(state: Mapping[str, str], *, audit: bool = False) -> List[str]:
    """`complete_task` 的前提，以及跑完时报的那一行。

    问的不是“人的套件过没过” —— 人的套件它看不见 —— 而是**它自己那套过没过**，这正是
    上游 manager 跑 `mix test` 的那一步。`audit` 在这里没有意义：仓库里不存在保留测试，
    保留的那套在仓库外面，由评分路径注入。保留这个形参只是为了和别的域同签名。

    `require_tests=True` 是因为这条路是**闸门**：一套空套件不算跑过。没有它，根 agent
    写了十六个实现文件、零个测试，也能喊 `complete_task` 并且被信 —— 这跟同一个类对每个
    子节点执行的规矩正好相反。
    """
    del audit
    return FLY.own_test_failures(state, require_tests=True)
reward = reward_test
HELD_OUT_FRAC = FLY.held_out_frac()


def llm_manager(complete):
    return _llm_manager(complete)


#: 失败时给执行者看什么。blind 域必须用这一套：需求、断言名、它报了什么，没有源码。
FAILURE = BLIND_FAILURE


def llm_executor(complete, *, editable=("**",), frozen=FROZEN):
    return _llm_executor(complete, editable=editable, frozen=frozen,
                         failure=BLIND_FAILURE)
