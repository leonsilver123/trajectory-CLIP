"""
tests/test_log_encoding.py - 日志语句里不许出现 GBK 无法编码的字符

## 为什么需要这条测试

本项目的日志走控制台，而 Windows 中文环境的控制台编码是 **cp936 (GBK)**。
`logging.StreamHandler.emit` 里是 `stream.write(msg + terminator)` —— 一旦消息里
有 GBK 编不出的字符（`⚠️`、`✓`、各种 emoji），就会抛：

    UnicodeEncodeError: 'gbk' codec can't encode character '\\u26a0'

**致命之处在于它不会让程序退出**：`logging` 默认吞掉异常并往 stderr 打一段
`--- Logging error ---` 的 traceback，然后**继续执行**。所以现象是：

- 日志文件里**没有那条消息**，只有一段看起来像崩溃的 traceback
- 程序照常往下跑，退出码正常
- 于是"告警没打出来"被误读成"没有告警"

2026-09-21 实测踩到：我给两个训练脚本加了"验证集口径不对"的运行时护栏，
两条护栏的消息都带 `⚠️` —— **两条护栏实际上都是哑的**，只在日志里留下一段
traceback。当时正是因为看到那段 traceback 才发现的。

记忆里其实存过这条坑（"GBK console can't print symbols"），但还是踩了，
所以改成测试来守。

## 这条测试怎么判

用 `ast` 解析（不是文本 grep）—— 文本扫描会漏掉跨行的日志调用，本次踩到的
两条护栏消息恰好都是跨行写法。只检查 `logger.<level>(...)` 调用的**字符串字面量
参数**，逐个字符试 `encode("gbk")`。

用 ASCII 标记代替：`[!]` / `[OK]` / `[x]`，而不是 `⚠️` / `✓` / `❌`。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_SCAN_DIRS = ("scripts", "src", "api")
_LOG_LEVELS = {"debug", "info", "warning", "error", "critical", "exception"}


def _iter_log_string_args():
    """
    遍历 `logger.<level>(...)` 与 `print(...)` 调用里的字符串字面量参数

    ⚠️ **`print` 必须一并覆盖。** 2026-09-21 的教训：本测试最初只扫 `logger.*`，
    结果 `scripts/eval_retrieval_hit_rate.py` 在**最后一行解读文字**里用了 `⇒`
    (U+21D2)，把整个评测脚本崩在收尾处 —— 表格已经打完了，却因为一行提示
    以非零码退出、JSON 也没写成。`print` 和 `logger` 走的是同一个 GBK 控制台，
    没有理由只查一个。

    用 ast 解析而非文本 grep：文本扫描会漏掉**跨行**的调用，而踩到的两处
    护栏消息恰好都是跨行写法。
    """
    for d in _SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for py in sorted(base.rglob("*.py")):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                is_log = (
                    isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "logger"
                    and fn.attr in _LOG_LEVELS
                )
                is_print = isinstance(fn, ast.Name) and fn.id == "print"
                if not (is_log or is_print):
                    continue
                for arg in node.args:
                    yield from _string_parts(py, node.lineno, arg)


def _string_parts(py, lineno, node):
    """
    从一个参数节点里抽字符串字面量，**f-string 也要抽**

    ⚠️ f-string 在 AST 里是 `ast.JoinedStr`（`values` 里既有 `Constant` 文本段、
    也有 `FormattedValue` 表达式），**不是** `ast.Constant`。
    2026-09-21 第二次踩到：扫描器最初只认 `ast.Constant`，于是**所有 f-string
    里的禁用字符全部漏检** —— `eval_chain_idf1.py` 的
    `f"...实测为 0 ⇒ ..."` 就是被漏掉的那一类，脚本崩在收尾处。
    f-string 恰恰是 print 里最常见的写法，漏掉它等于没扫。
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield py, lineno, node.value
    elif isinstance(node, ast.JoinedStr):
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                yield py, lineno, part.value


def test_no_unencodable_chars_in_log_messages():
    """
    日志消息必须能被 cp936 编码

    失败时给出的路径:行号就是出问题的那条日志语句。
    """
    offenders = []
    for py, lineno, text in _iter_log_string_args():
        for ch in text:
            try:
                ch.encode("gbk")
            except UnicodeEncodeError:
                rel = py.relative_to(ROOT)
                offenders.append(
                    f"{rel}:{lineno} 字符 {ch!r} (U+{ord(ch):04X}) —— {text[:50]!r}"
                )
                break

    assert not offenders, (
        "以下日志语句含 GBK(cp936) 无法编码的字符。这类消息**不会让程序崩溃**，"
        "而是被 logging 吞掉、只在日志里留下一段 'Logging error' traceback —— "
        "于是告警看起来像没触发。请改用 ASCII 标记（[!] / [OK] / [x]）：\n  "
        + "\n  ".join(offenders)
    )


def test_scanner_actually_finds_log_calls():
    """
    反向验证：扫描器本身要能扫到东西

    若哪天日志调用改成了别的名字（或 ast 逻辑写错），上面那条会**空过** ——
    永远通过、什么也没守住。这条用来发现"扫描器失效"。
    """
    found = list(_iter_log_string_args())
    assert len(found) > 100, (
        f"只扫到 {len(found)} 条日志字符串参数，数量异常偏低 —— "
        "扫描逻辑可能已经失效，上一条测试会在空过的情况下假装通过"
    )


@pytest.mark.parametrize("text", ["⚠", "️", "✓", "❌", "✗"])
def test_known_offenders_really_are_unencodable(text):
    """
    钉住"这几个字符确实编不出来"这个前提

    若将来控制台编码不再是 cp936（例如改成 UTF-8），本文件的理由就不再成立，
    这几条会失败并提示重新评估 —— 而不是让一条过时的规则继续拦人。

    注意**不要**想当然地往这个列表里加符号。实测（见下表）GBK **能**编码
    `→`(U+2192) `←` `●` `★` `√` `×`，却编不出 `✓`(U+2713) `✗` `✘`。
    我最初把 `→` 也列进来，测试直接失败 —— 这条用例的价值就在这儿：
    它拦住的是"把能用的符号也一起禁掉"。
    """
    with pytest.raises(UnicodeEncodeError):
        text.encode("gbk")
