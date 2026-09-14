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

#: 需求和驱动。人写的全部，拒绝给任何提案，评分前恢复原样。
FROZEN = ("REQUIREMENTS.md", "fly.py")

#: 推进每一个 brief 的东西，按这个顺序。
CONTRACTS = ("REQUIREMENTS.md", "fly.py")

SCORING = "黑盒验收（agent 读不到），外加一套开跑前封存的事后测量"
CASE_NOUN = "验收断言"
GROUP_NOUN = "验收文件"

ENTRY = "src/__init__.py"

#: 没有参考实现，也就没有离线规则化 actor；写一个出来就等于把该生长的设计先写了。
REQUIRES_MODEL = True

OBJECTIVE = (
    "实现 `REQUIREMENTS.md` 要的那个软件：一个根在 `src/` 的 Python 包，把一只果蝇的大脑"
    "照真实连接组建出来，让它在竞技场里执行任务、领取奖惩、并因此改变行为，再把整个东西"
    "作为一个网页端出来。冻结的 `fly.py` 只 import `src` 的四个名字 —— `make_brain`、"
    "`train`、`run_episode`、`handle` —— 那四个就是公开接口。`src/` 以下怎么分层、每层叫"
    "什么、谁调用谁，全部由你们决定。可以用 numpy 和其它第三方包。"
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
    'acceptance/test_api.py': r'''"""用户验收：只通过驱动用到的四个名字，黑盒地问"我要的东西能用吗"。

这些断言从不进入仓库。agent 看到的是它们的名字和报错，看不到一行源码 ——
仓库里的每一个测试都是 agent 自己写的。
"""

import json

from src import handle, make_brain, run_episode, train

TASKS = ("taxis", "avoid", "choice")


def rows(task, episodes=4, seed=0):
    return train(task, episodes, seed)["episodes"]


def test_the_api_reports_where_the_animal_is_and_what_its_brain_is_doing():
    status, content_type, body = handle("GET", "/api/state")
    assert status == 200 and "json" in content_type
    state = json.loads(body)
    assert "position" in state and len(state["position"]) == 2
    assert isinstance(state.get("brain"), dict) and len(state["brain"]) >= 4


def test_the_api_can_step_reset_and_train():
    handle("POST", "/api/reset", json.dumps({"task": "avoid", "seed": 0}))
    before = json.loads(handle("GET", "/api/state")[2])["step"]
    handle("POST", "/api/step", json.dumps({"steps": 10}))
    after = json.loads(handle("GET", "/api/state")[2])["step"]
    assert after > before
    status, _, _ = handle("POST", "/api/train", json.dumps({"task": "avoid",
                                                            "episodes": 2}))
    assert status == 200
    assert json.loads(handle("POST", "/api/reset",
                             json.dumps({"task": "avoid", "seed": 0}))[2])["step"] == 0


def test_an_unknown_route_is_refused_and_a_bad_body_is_too():
    assert handle("GET", "/api/nothing-here")[0] == 404
    assert handle("POST", "/api/step", "{not json")[0] == 400
''',
    'acceptance/test_learning.py': r'''"""用户验收：只通过驱动用到的四个名字，黑盒地问"我要的东西能用吗"。

这些断言从不进入仓库。agent 看到的是它们的名字和报错，看不到一行源码 ——
仓库里的每一个测试都是 agent 自己写的。
"""

import json

from src import handle, make_brain, run_episode, train

TASKS = ("taxis", "avoid", "choice")


def rows(task, episodes=4, seed=0):
    return train(task, episodes, seed)["episodes"]


def test_a_fresh_animal_is_punished_in_the_avoidance_assay():
    """没学过的动物会一头撞进惩罚区 —— 否则这个实验什么也测不到。"""
    naive = make_brain({"seed": 0})
    assert sum(run_episode(naive, "avoid", 1000 + i, learning=False)["shocks"]
               for i in range(3)) > 0


def test_training_reduces_the_punishment_the_animal_takes():
    shocks = [r["shocks"] for r in rows("avoid", 12)]
    assert sum(shocks[:3]) > 0
    assert sum(shocks[-3:]) < sum(shocks[:3])


def test_what_was_learned_transfers_to_arenas_it_never_saw():
    """关掉可塑性、用没训练过的种子 —— 唯一诚实的学习度量。"""
    trained = train("avoid", 10, 0)["brain"]
    naive = make_brain({"seed": 0})
    unseen = range(2000, 2005)
    before = sum(run_episode(naive, "avoid", s, learning=False)["shocks"] for s in unseen)
    after = sum(run_episode(trained, "avoid", s, learning=False)["shocks"] for s in unseen)
    assert before > 0 and after < before


def test_an_episode_with_learning_off_leaves_the_animal_unchanged():
    brain = make_brain({"seed": 0})
    for s in range(3):
        run_episode(brain, "avoid", s, learning=False)
    fresh = make_brain({"seed": 0})
    before = [run_episode(fresh, "avoid", 3000 + i, learning=False)["shocks"]
              for i in range(3)]
    after = [run_episode(brain, "avoid", 3000 + i, learning=False)["shocks"]
             for i in range(3)]
    assert before == after
''',
    'acceptance/test_page.py': r'''"""用户验收：只通过驱动用到的四个名字，黑盒地问"我要的东西能用吗"。

这些断言从不进入仓库。agent 看到的是它们的名字和报错，看不到一行源码 ——
仓库里的每一个测试都是 agent 自己写的。
"""

import json

from src import handle, make_brain, run_episode, train

TASKS = ("taxis", "avoid", "choice")


def rows(task, episodes=4, seed=0):
    return train(task, episodes, seed)["episodes"]


def test_the_page_is_served_at_the_root():
    status, content_type, body = handle("GET", "/")
    assert status == 200 and "html" in content_type.lower()
    assert body.lstrip().lower().startswith("<!doctype html")


def test_the_page_shows_the_arena_the_brain_and_the_learning_curve():
    _, _, page = handle("GET", "/")
    low = page.lower()
    assert "canvas" in low
    for word in ("arena", "brain", "learn"):
        assert word in low, word
''',
    'acceptance/test_training.py': r'''"""用户验收：只通过驱动用到的四个名字，黑盒地问"我要的东西能用吗"。

这些断言从不进入仓库。agent 看到的是它们的名字和报错，看不到一行源码 ——
仓库里的每一个测试都是 agent 自己写的。
"""

import json

from src import handle, make_brain, run_episode, train

TASKS = ("taxis", "avoid", "choice")


def rows(task, episodes=4, seed=0):
    return train(task, episodes, seed)["episodes"]


def test_every_task_can_be_trained():
    for task in TASKS:
        assert len(rows(task, 2)) == 2, task


def test_each_episode_reports_its_shocks_reward_and_length():
    for i, row in enumerate(rows("avoid", 3)):
        assert row["episode"] == i
        assert isinstance(row["shocks"], (int, float)) and row["shocks"] >= 0
        assert isinstance(row["reward"], (int, float))
        assert row["steps"] > 0


def test_training_returns_the_animal_it_trained():
    result = train("avoid", 3, 0)
    assert result["brain"] is not None
    assert run_episode(result["brain"], "avoid", 7, learning=False)["shocks"] >= 0


def test_the_same_seed_gives_the_same_run():
    assert [r["shocks"] for r in rows("avoid", 5)] == [r["shocks"] for r in rows("avoid", 5)]
''',
}

#: 封存的事后测量。开跑前写死，全程不进仓库，跑完只用来报一个数。
_SEALED = {
    'sealed/test_sealed.py': r'''"""封存的事后测量：写在开跑之前，全程不进仓库，任何 agent 读不到，跑完只用来报一个数。

这是上游的协议 —— Genesis 拿 c-testsuite、LLVM、Csmith 验证，那些是实验者事后量的外部
基准，不是 agent 的内循环。它们和 `acceptance/` 问的是同一批要求，但换了问法、换了种子、
换了任务，所以"验收全过"和"封存套件全过"之间的差距，就是它把东西做对了还是只对着验收写的
度量。

同样只用驱动那四个名字。有一条要求是这套测不了的：**大脑是不是真的照连接组建的**。黑盒
问不出来，只能靠父节点评审读 diff、和人读结果。这条限制写在这里，不藏着。
"""

import json

from src import handle, make_brain, run_episode, train


def test_learning_shows_up_in_the_choice_assay_too():
    trained = train("choice", 10, 1)["brain"]
    naive = make_brain({"seed": 1})
    unseen = range(4000, 4006)
    before = sum(run_episode(naive, "choice", s, learning=False)["shocks"] for s in unseen)
    after = sum(run_episode(trained, "choice", s, learning=False)["shocks"] for s in unseen)
    assert before > 0 and after < before


def test_the_gain_from_training_survives_a_different_seed():
    trained = train("avoid", 10, 5)["brain"]
    naive = make_brain({"seed": 5})
    unseen = range(5000, 5006)
    before = sum(run_episode(naive, "avoid", s, learning=False)["shocks"] for s in unseen)
    after = sum(run_episode(trained, "avoid", s, learning=False)["shocks"] for s in unseen)
    assert after < 0.6 * before


def test_more_training_is_not_worse_than_less():
    short = train("avoid", 4, 2)["brain"]
    long = train("avoid", 16, 2)["brain"]
    unseen = range(6000, 6006)
    a = sum(run_episode(short, "avoid", s, learning=False)["shocks"] for s in unseen)
    b = sum(run_episode(long, "avoid", s, learning=False)["shocks"] for s in unseen)
    assert b <= a


def test_an_animal_trained_on_one_task_still_works_on_another():
    """不应该为了通过一个实验把另一个弄坏。"""
    trained = train("avoid", 10, 3)["brain"]
    assert run_episode(trained, "taxis", 7000, learning=False)["steps"] > 0


def test_two_animals_with_different_seeds_are_not_the_same_animal():
    one = [run_episode(make_brain({"seed": 11}), "avoid", 8000 + i, learning=False)["shocks"]
           for i in range(4)]
    two = [run_episode(make_brain({"seed": 22}), "avoid", 8000 + i, learning=False)["shocks"]
           for i in range(4)]
    assert one != two


def test_the_taxis_assay_ends_sooner_than_it_times_out():
    """趋向实验要真的能到达 —— 每局都跑满上限说明动物根本没找到源。"""
    steps = [r["steps"] for r in train("taxis", 6, 0)["episodes"]]
    assert min(steps) < max(steps) or min(steps) < 400


def test_the_state_the_page_draws_changes_as_the_animal_moves():
    handle("POST", "/api/reset", json.dumps({"task": "taxis", "seed": 0}))
    first = json.loads(handle("GET", "/api/state")[2])
    handle("POST", "/api/step", json.dumps({"steps": 25}))
    later = json.loads(handle("GET", "/api/state")[2])
    assert first["position"] != later["position"]


def test_the_page_exposes_more_than_one_layer_of_the_brain():
    """需求里写了"要能看见大脑各层此刻的活动"，所以状态里不能只有一个标量。"""
    handle("POST", "/api/reset", json.dumps({"task": "avoid", "seed": 0}))
    handle("POST", "/api/step", json.dumps({"steps": 5}))
    brain = json.loads(handle("GET", "/api/state")[2])["brain"]
    populations = [k for k, v in brain.items() if isinstance(v, (list, dict)) and v]
    assert len(populations) >= 3, sorted(brain)


def test_a_new_brain_through_the_api_has_forgotten_everything():
    handle("POST", "/api/reset", json.dumps({"task": "avoid", "seed": 0, "brain": "new"}))
    handle("POST", "/api/train", json.dumps({"task": "avoid", "episodes": 6}))
    handle("POST", "/api/reset", json.dumps({"task": "avoid", "seed": 0, "brain": "new"}))
    handle("POST", "/api/step", json.dumps({"steps": 1}))
    fresh = json.loads(handle("GET", "/api/state")[2])
    assert fresh["step"] >= 1


def test_the_driver_is_reproducible_across_two_calls_in_one_process():
    one = [r["reward"] for r in train("choice", 5, 9)["episodes"]]
    two = [r["reward"] for r in train("choice", 5, 9)["episodes"]]
    assert one == two
''',
}

#: 用户的要求，原话。
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

## 四、怎么算做完

`fly.py` 是冻结的，你们不能改它。它就是这份需求的可执行形式 —— 我要的是一个**能跑的程序**，
不是一个没人能调用的包：

```
python fly.py train --task avoid --episodes 20     # 训练，打出每集的电击数
python fly.py probe --task avoid --trials 5        # 关掉可塑性，朴素 vs 训练过
python fly.py serve --port 8000                    # 起服务，浏览器打开
```

它只通过 `src` 这一个包调用你们写的东西。`fly.py` 里 import 了哪些名字，那些就是公开接口，
其余全是你们的内部事务。

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

#: 那份需求的可执行形式。冻结：它是需求，不是实现。
_DRIVER = r'''#!/usr/bin/env python3
"""fly -- 一只果蝇，一个大脑，和一个能看着它学习的网页。

这个文件是**冻结的**：没有 agent 可以写它。它是 `REQUIREMENTS.md` 的可执行形式 ——
我要的是一个能跑的程序，而不是一个没人能调用的包。

    python fly.py train --task avoid --episodes 20
    python fly.py probe --task avoid --episodes 10 --trials 5
    python fly.py serve --port 8000

它只通过 `src` 这一个包调用你们写的东西，并且**只用到下面四个名字**。其余一切 ——
分几层、每层叫什么、谁调用谁、写哪些测试 —— 都是你们的内部事务。

    make_brain(config=None) -> brain
        一只没有任何经验的动物。`config` 是个字典，至少认得 `seed`。

    train(task, episodes, seed) -> {"episodes": [...], "brain": brain}
        跑 `episodes` 局，每局一行，按顺序放在 "episodes" 里。每行至少要有
        `episode`（从 0 数）、`shocks`（这一局挨了多少次惩罚）、`reward`、`steps`。
        "brain" 是训练完的那只动物。

    run_episode(brain, task, seed, learning=True) -> {"shocks": ..., ...}
        用给定的动物跑一局。`learning=False` 时不得改变它 —— 这是唯一能诚实
        比较"学过"和"没学过"的办法。

    handle(method, path, body=None) -> (status, content_type, body)
        HTTP 的全部逻辑，一个纯函数，不开 socket。下面那层 http.server 是我写的。
        `GET /` 必须返回那个网页。

任务名至少要有 `taxis`、`avoid`、`choice` 三个。
"""

import argparse
import sys


def cmd_train(args):
    from src import train
    rows = train(args.task, args.episodes, args.seed)["episodes"]
    print(f"{'ep':>4} {'shocks':>7} {'reward':>9} {'steps':>6}")
    for row in rows:
        print(f"{row['episode']:>4} {row['shocks']:>7} {row['reward']:>9.2f} "
              f"{row['steps']:>6}")
    head = sum(r["shocks"] for r in rows[:3]) / max(1, len(rows[:3]))
    tail = sum(r["shocks"] for r in rows[-3:]) / max(1, len(rows[-3:]))
    print(f"\n前三局平均 {head:.1f} 次惩罚，后三局 {tail:.1f}")
    return 0


def cmd_probe(args):
    """训练一只，然后把它和一只没学过的放在同样的、没见过的竞技场里，关掉可塑性。

    这是唯一诚实的学习度量：同样的环境、没训练过的种子、测试期间什么都不改变。
    """
    from src import make_brain, run_episode, train
    trained = train(args.task, args.episodes, args.seed)["brain"]
    naive = make_brain({"seed": args.seed})
    totals = {}
    for label, brain in (("naive", naive), ("trained", trained)):
        shocks = [run_episode(brain, args.task, 1000 + i, learning=False)["shocks"]
                  for i in range(args.trials)]
        totals[label] = sum(shocks)
        print(f"{label:>8}: {sum(shocks):>5} 次惩罚 / {args.trials} 局  {shocks}")
    print(f"\n学习带来的减少：{totals['naive'] - totals['trained']}")
    return 0


def cmd_serve(args):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from src import handle

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, method):
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length).decode("utf-8") if length else None
            status, content_type, payload = handle(method, self.path, body)
            data = payload.encode("utf-8") if isinstance(payload, str) else payload
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._respond("GET")

        def do_POST(self):
            self._respond("POST")

        def log_message(self, *a):
            pass

    server = HTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}/")
    sys.stdout.flush()
    server.serve_forever()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("train", cmd_train), ("probe", cmd_probe), ("serve", cmd_serve)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if name in ("train", "probe"):
            p.add_argument("--task", default="avoid")
            p.add_argument("--episodes", type=int, default=20)
            p.add_argument("--seed", type=int, default=0)
        if name == "probe":
            p.add_argument("--trials", type=int, default=5)
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def initial_files() -> Dict[str, str]:
    """整个仓库的起点：一份需求、一个驱动、一份写模块的须知。没有测试，没有记录。"""
    return {"REQUIREMENTS.md": _REQUIREMENTS, "fly.py": _DRIVER,
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
    """
    del audit
    return FLY.own_test_failures(state)
reward = reward_test
HELD_OUT_FRAC = FLY.held_out_frac()


def llm_manager(complete):
    return _llm_manager(complete)


#: 失败时给执行者看什么。blind 域必须用这一套：需求、断言名、它报了什么，没有源码。
FAILURE = BLIND_FAILURE


def llm_executor(complete, *, editable=("**",), frozen=FROZEN):
    return _llm_executor(complete, editable=editable, frozen=frozen,
                         failure=BLIND_FAILURE)
