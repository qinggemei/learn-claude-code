#!/usr/bin/env python3
"""
s03_permission.py - Permission System

Three gates inserted before tool execution:

    Gate 1: Hard deny list (rm -rf /, sudo, ...)
    Gate 2: Rule matching (write outside workspace? destructive cmd?)
    Gate 3: User approval (pause and wait for confirmation)

    +-------+    +--------+    +--------+    +--------+    +------+
    | Tool  | -> | Gate 1 | -> | Gate 2 | -> | Gate 3 | -> | Exec |
    | call  |    | deny?  |    | match? |    | allow? |    |      |
    +-------+    +--------+    +--------+    +--------+    +------+
         |            |             |             |
         v            v             v             v
      (normal)     (blocked)    (ask user)   (user says no?)

Only one line added to the agent loop:

    if not check_permission(block):
        continue

Builds on s02 (multi-tool). Usage:

    python s03_permission/code.py
    Needs: pip install anthropic python-dotenv + ANTHROPIC_API_KEY in .env
"""

import os, subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

load_dotenv(override=True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

MODEL = os.getenv("OPENAI_MODEL")

WORKDIR = Path.cwd()
SYSTEM = f"You are a coding agent at {WORKDIR}. 所有破坏性操作都需要用户批准。"


# s02_tool_use


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@tool
def run_read(path, limit=None):
    """
    Read file contents.
    """
    lines = safe_path(path).read_text(encoding="utf-8").splitlines()
    if limit:
        lines = lines[:limit]
    return "\n".join(lines)


@tool
def run_write(path, content):
    """
    Write content to file.
    """
    safe_path(path).write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


@tool
def run_edit(path, old_text, new_text):
    """
    Replace text in file once.
    """
    text = safe_path(path).read_text(encoding="utf-8")
    if old_text not in text:
        return "Error: text not found"
    safe_path(path).write_text(text.replace(old_text, new_text, 1))
    return f"Edited {path}"


@tool
def run_glob(pattern):
    """
    Find files by pattern.
    """
    import glob as g
    return "\n".join(g.glob(pattern, root_dir=WORKDIR))


@tool
def bash(command: str) -> str:
    """
    Run a shell command.
    """
    dangerous = ["rm -rf /", "del", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


tools = [bash, run_read, run_write, run_edit, run_glob]

tool_dict = {tool.name: tool for tool in tools}

model = init_chat_model(
    model=MODEL,
    model_provider="openai"
)

model_with_tools = model.bind_tools(tools)

# ═══════════════════════════════════════════════════════════
#  NEW in s03: Three-Gate Permission Pipeline
# ═══════════════════════════════════════════════════════════

# Gate 1: Hard deny list — always forbidden
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]


def check_deny_list(command: str) -> str | None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None


# Gate 2: Rule matching — context-dependent checks
PERMISSION_RULES = [
    {"tools": ["write_file", "edit_file"],
     "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
     "message": "Writing outside workspace"},
    {"tools": ["bash"],
     "check": lambda args: any(kw in args.get("command", "") for kw in ["rm ", "del", "> /etc/", "chmod 777"]),
     "message": "Potentially destructive command"},
]


def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


# Gate 3: User approval — wait for confirmation after rule match
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print("询问用户是否愿意：")
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] :").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


def check_permission(tool_call):
    print("进入权限校验")
    if tool_call["name"] == "bash":
        reason = check_deny_list(tool_call["args"]['command'])
        if reason:
            print(f"\n\033[31m⛔ {reason}\033[0m")
            return False

    reason = check_rules(tool_call["name"], tool_call["args"])
    if reason:
        choice = ask_user(tool_call["name"], tool_call["args"], reason)
        if choice == "deny":
            return False

    return True


def agent_loop(messages: list):
    while True:
        # 返回的是工具名与参数 / 返回消息
        # AiMessage
        response = model_with_tools.invoke(messages)

        print("start------------ model_with_tools -------------------")
        print(response)
        print("----------------- model_with_tools ----------------end")

        messages.append(response)

        if (len(response.tool_calls) == 0):
            return

        for tool_call in response.tool_calls:
            # 打印 入参
            print(f"调用工具：{tool_call['name']}")
            print(f"\033[33m$ {tool_call["args"]}\033[0m")
            tool = tool_dict[tool_call["name"]]

            # check permission
            if not check_permission(tool_call):
                messages.append(ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="用户不同意进行这个操作"
                ))
                continue

            # 调用工具并返回结果
            # ToolMessage
            tool_response = tool.invoke(tool_call)
            # 工具返回结果
            print(tool_response.content)
            # messages.append(tool_response)
            # 添加到历史记录中
            messages.append(tool_response)


if __name__ == "__main__":
    print("s01: Agent Loop (langchain)")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    history.append(AIMessage(SYSTEM))
    while True:
        try:
            query = input("s03 >> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append(HumanMessage(query))
        agent_loop(history)
        print("start--------------------------")
        print(history[-1].content)
        print("----------------------------end")

# 创建test.txt
# 删除test.txt
