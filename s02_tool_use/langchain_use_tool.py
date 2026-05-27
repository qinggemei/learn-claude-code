# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
s02: Tool Use — 在 s01 基础上新增 4 个工具 + 分发映射。

运行: python s02_tool_use/code.py
需要: pip install anthropic python-dotenv + .env 中配置 ANTHROPIC_API_KEY

本文件 = s01 的全部代码 + 以下新增:
  + run_read / run_write / run_edit / run_glob 四个工具实现
  + TOOL_HANDLERS 分发映射（替代 s01 中硬编码的 run_bash 调用）
  + safe_path 路径安全校验

循环本身（agent_loop）与 s01 完全一致。
"""
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

load_dotenv(override=True)

os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL"] = os.getenv("OPENAI_MODEL")

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


model = init_chat_model(
    model=os.getenv("OPENAI_MODEL"),  # 模型名称，这里选择qwen3.5-plus，这是一个多模态模型，支持图片、文本、音频、视频
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ═══════════════════════════════════════════════════════════
#  NEW in s02: 4 个新工具
# ═══════════════════════════════════════════════════════════

WORKDIR = Path.cwd()

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
    safe_path(path).write_text(content,encoding="utf-8")
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
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
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

model_with_tools = model.bind_tools(tools)

def agent_loop(messages: list):
    while True:
        # 返回的是工具名与参数
        response = model_with_tools.invoke(messages)

        messages.append(response)

        if (len(response.tool_calls) == 0):
            return

        for tool_call in response.tool_calls:
            # 打印 入参
            print("tool name:", tool_call["name"])
            print("args:")
            print(tool_call)
            if len(tool_call["args"])>0:
                print(f"\033[33m$ {tool_call["args"].__str__()}\033[0m")
            tool = tool_dict[tool_call["name"]]
            # 调用工具并返回结果
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
            query = input("s01 >> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append(HumanMessage(query))
        agent_loop(history)
        print("start--------------------------")
        print(history[-1].content)
        print("----------------------------end")

