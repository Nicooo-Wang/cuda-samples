#!/usr/bin/env python3
"""从各教程目录的 Makefile 生成 compile_commands.json（供 VS Code / clangd 跳转用）。

原理：不真正编译，只用 `make -B -n` 让 make 把命令**打印**出来
（-n = dry run，-B = 假装所有目标都过期，否则已经编译好的目录什么都不打印），
再把打印出来的 nvcc 命令行翻译成 compile_commands.json 的条目。

好处是 flag 永远和 Makefile 保持一致：以后往 Makefile 里加 -I 或改 -arch，
重跑一次这个脚本就行，不用手动同步第二份配置。

用法：
    python3 tools/gen_compile_commands.py          # 写到仓库根的 compile_commands.json
    python3 tools/gen_compile_commands.py -o /tmp/x.json
    python3 tools/gen_compile_commands.py -q       # 只在内容真变了时打一行（给 Makefile 钩子用）

各教程的 Makefile 末尾 include 了 tools/compile_commands.mk，所以正常 `make` 的时候
这个脚本会自动跑一遍，不用手动记着刷新。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 这些目录里的 Makefile 不是我们的教程，跳过
SKIP_DIRS = {".git", ".venv", ".claude", "third-party", "node_modules", "__pycache__"}

# 只有这些后缀算翻译单元（compile_commands.json 的 "file" 字段）
SOURCE_SUFFIXES = {".cu", ".cpp", ".cc", ".cxx", ".c"}

# 链接期才用得到的 flag，对 IntelliSense 没有意义，顺手滤掉让条目干净些
LINK_ONLY_PREFIXES = ("-l", "-L")

# 递归刹车。
# 教程的 Makefile 会 include tools/compile_commands.mk，那个 .mk 在解析期就调本脚本；
# 而本脚本又要去各目录跑 `make -B -n`，那些 make 同样会解析到 .mk ——
# 不刹一脚就是 11^n 级别的自我调用。所以往子 make 的环境里塞这个变量，
# .mk 看见它就整段跳过。
GUARD_ENV = "CUDA_COURSE_CC_GEN"


def find_makefile_dirs(root: Path) -> list[Path]:
    """找出所有带 Makefile 的教程目录。"""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        # 原地裁剪 dirnames，os.walk 就不会下去了
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if "Makefile" in filenames or "makefile" in filenames:
            found.append(Path(dirpath))
    return sorted(found)


def dry_run_make(directory: Path) -> list[str]:
    """在 directory 里跑 `make -B -n`，返回打印出来的命令行列表。"""
    env = dict(os.environ, **{GUARD_ENV: "1"})
    try:
        proc = subprocess.run(
            ["make", "-B", "-n", "all"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  ! {directory}: make 跑不起来 ({exc})", file=sys.stderr)
        return []

    if proc.returncode != 0:
        # 没有 all 目标的话退回默认目标再试一次
        proc = subprocess.run(
            ["make", "-B", "-n"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    if proc.returncode != 0:
        print(f"  ! {directory}: make -n 失败：{proc.stderr.strip()[:200]}", file=sys.stderr)
        return []

    return proc.stdout.splitlines()


def is_compiler(token: str) -> bool:
    """token 看起来像编译器驱动吗（nvcc / gcc / g++ / clang++ ...）。"""
    name = Path(token).name
    return name in {"nvcc", "gcc", "g++", "cc", "c++", "clang", "clang++"}


def resolve_compiler(token: str) -> str:
    """把 `nvcc` 解析成绝对路径。

    cpptools 靠 basename 匹配 ^nvcc$ 来识别 CUDA 编译器，所以解析成
    /usr/local/cuda/bin/nvcc 既保留了 basename，又让它不依赖 VS Code 继承到的 PATH
    （GUI 起的 VS Code 往往没有 shell 里那份 PATH）。
    """
    if os.path.isabs(token):
        return token
    which = shutil.which(token)
    return which if which else token


def parse_command(line: str, directory: Path) -> dict | None:
    """把一行 shell 命令翻译成一条 compile_commands 条目，认不出来就返回 None。"""
    line = line.strip()
    if not line or line.startswith(("@", "#", "echo", "if ", "for ", "rm ")):
        return None

    try:
        tokens = shlex.split(line)
    except ValueError:
        return None
    if not tokens or not is_compiler(tokens[0]):
        return None

    arguments: list[str] = [resolve_compiler(tokens[0])]
    source: str | None = None

    i = 1
    while i < len(tokens):
        tok = tokens[i]

        # -o <output>：IntelliSense 不需要产物路径，丢掉
        if tok == "-o":
            i += 2
            continue
        if tok.startswith("-o") and len(tok) > 2:
            i += 1
            continue

        # 链接期 flag，丢掉
        if tok.startswith(LINK_ONLY_PREFIXES) and not tok.startswith("-lineinfo"):
            i += 1
            continue

        # 源文件：记下来，同时保留在 arguments 里（compile_commands 的惯例）
        if not tok.startswith("-") and Path(tok).suffix in SOURCE_SUFFIXES:
            source = tok
            arguments.append(tok)
            i += 1
            continue

        arguments.append(tok)
        i += 1

    if source is None:
        return None

    return {
        "directory": str(directory),
        "file": str((directory / source).resolve()),
        "arguments": arguments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-o",
        "--output",
        default=str(REPO_ROOT / "compile_commands.json"),
        help="输出路径（默认：仓库根的 compile_commands.json）",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="安静模式：不打每个目录的明细，内容没变时一个字都不打（给 Makefile 钩子用）",
    )
    args = parser.parse_args()

    if shutil.which("make") is None:
        # 给 Makefile 钩子用时不能因为环境缺 make 就让 build 挂掉
        print("找不到 make，没法抓编译命令", file=sys.stderr)
        return 0 if args.quiet else 1

    entries: list[dict] = []
    seen: set[str] = set()

    for directory in find_makefile_dirs(REPO_ROOT):
        rel = directory.relative_to(REPO_ROOT)
        lines = dry_run_make(directory)
        count = 0
        for line in lines:
            entry = parse_command(line, directory)
            if entry is None:
                continue
            if entry["file"] in seen:
                continue
            seen.add(entry["file"])
            entries.append(entry)
            count += 1
        if not args.quiet:
            print(f"  {rel}: {count} 个翻译单元")

    if not entries:
        print("一条编译命令都没抓到，检查各目录的 Makefile", file=sys.stderr)
        return 0 if args.quiet else 1

    out = Path(args.output)
    payload = json.dumps(entries, indent=2, ensure_ascii=False) + "\n"

    # 内容没变就别写。
    # cpptools 监听这个文件的 mtime，一写就重扫一遍数据库；
    # 挂到 Makefile 上以后每次 make 都会走到这儿，无脑重写会让 IntelliSense
    # 每编译一次抖一下。
    if out.exists() and out.read_text(encoding="utf-8") == payload:
        if not args.quiet:
            print(f"\n{out} 无变化（{len(entries)} 条）")
        return 0

    out.write_text(payload, encoding="utf-8")
    print(f"\n写入 {out}（{len(entries)} 条）" if not args.quiet
          else f"[compile_commands] 已更新（{len(entries)} 条翻译单元）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
