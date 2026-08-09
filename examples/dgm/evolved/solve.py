"""The agent: read a failing task, edit `lib.py` to fix it."""
import pathlib
import re
from tools import read_source, run_tests, write_source


def _strip_code_fences(reply):
    text = reply.strip()
    if not text:
        return ""
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        out = []
        for line in lines:
            if line.strip().startswith("```"):
                break
            out.append(line)
        if out:
            return "\n".join(out).strip() + "\n"
    match = re.search(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"
    return text + "\n"


def _is_success(out):
    return " passed" in out and " failed" not in out and " error" not in out


def solve(task_dir, llm):
    """Return nothing; edit files in `task_dir` in place.

    `llm(prompt) -> str` is the only model access. `run_tests(task_dir)`
    returns (passed, failed, output).
    """
    task_path = pathlib.Path(task_dir)
    source = read_source(task_dir)

    try:
        test_source = (task_path / "test_lib.py").read_text()
    except FileNotFoundError:
        test_source = ""

    prompt_head = (
        "Fix the bug so every test passes. Reply with the complete new "
        "contents of lib.py and nothing else -- no fences, no commentary.\n\n"
        f"lib.py:\n{source}\n\n"
    )
    if test_source:
        prompt_head += f"test_lib.py:\n{test_source}\n\n"

    _, _, failure = run_tests(task_dir)
    prompt = prompt_head + f"pytest says:\n{failure}"

    for _ in range(3):
        reply = llm(prompt)
        code = _strip_code_fences(reply)
        write_source(task_dir, code)

        try:
            compile(code, str(task_path / "lib.py"), "exec")
        except SyntaxError as exc:
            prompt = prompt_head + (
                "Your previous reply had invalid Python. pytest says:\n"
                f"SyntaxError in submitted lib.py:\n{exc}"
            )
            continue

        _, _, out = run_tests(task_dir)
        if _is_success(out):
            return

        prompt = prompt_head + (
            "Your previous attempt did not pass all tests. "
            "Reply again with corrected complete lib.py and nothing else.\n\n"
            f"pytest says:\n{out}"
        )