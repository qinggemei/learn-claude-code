import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_core.tools import tool

from s04_hooks.s04_code import DESTRUCTIVE

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
def read_file(path, limit=None):
    """
    Read file contents.
    """
    lines = safe_path(path).read_text(encoding="utf-8").splitlines()
    if limit:
        lines = lines[:limit]
    return "\n".join(lines)


@tool
def write_file(path, content):
    """
    Write content to file.
    """
    safe_path(path).write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


@tool
def edit_file(path, old_text, new_text):
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


tools = [bash, read_file, write_file, edit_file, run_glob]

tool_dict = {tool.name: tool for tool in tools}

# s04 hook

HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


model = init_chat_model(
    model=MODEL,
    model_provider="openai"
)

model_with_tools = model.bind_tools(tools)

# s03 Permission

# 拒绝操作
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]
DESTRUCTIVE = ["rm", "> /etc/", "chmod 777"]


def permission_hook(tool_call):
    if tool_call["name"] == "bash":
        for pattern in DENY_LIST:
            if pattern in tool_call["args"]["command"]:
                print(f"\n\033[31m⛔ Blocked: '{pattern}'\033[0m")
                return "Permission denied by deny list"
        for kw in DESTRUCTIVE:
            if kw in tool_call["args"]["command"]:
                print(f"\n\033[33m⚠  Potentially destructive command\033[0m")
                print(f"   Tool: {tool_call["name"]}({tool_call["args"]})")
    if tool_call["name"] in ["write_file", "edit_file"]:
        path = tool_call["args"]["path"]
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m⚠  Writing outside workspace\033[0m")
            print(f"   Tool: {tool_call["name"]}({tool_call["args"]})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None

def log_hook(tool_call):
    """PreToolUse: log every tool call."""
    args_preview = str(list(tool_call["args"])[:2])[:60]
    print(f"\033[90m[HOOK] {tool_call["name"]}({args_preview})\033[0m")
    return None

def large_output_hook(tool_call,output):
    print("output")
    print(output)
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] ⚠ Large output from {tool_call["name"]}: {len(str(output))} chars\033[0m")
    return None

def context_inject_hook(query: str):
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None

def summary_hook(messages: list):
    tool_count = sum(1 for m in messages
                     for b in (m.content if isinstance(m.content, list) else [])
                     if isinstance(b, dict) and b["type"] == "tool")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)



def agent_loop(messages: list):
    while True:
        # 返回的是工具名与参数 / 消息(AIMessage)
        response = model_with_tools.invoke(messages)
        messages.append(response)

        # 没有工具调用
        if len(response.tool_calls) == 0:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        for tool_call in response.tool_calls:
            # 打印 入参
            print(f"调用工具：{tool_call['name']}，args：{tool_call['args']}")

            blocked = trigger_hooks("PreToolUse", tool_call)
            if blocked:
                continue

            tool = tool_dict.get(tool_call["name"])

            tool_response = tool.invoke(tool_call)
            messages.append(tool_response)
            trigger_hooks("PostToolUse", tool_call, tool_response.content)  # s04: post hook



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
