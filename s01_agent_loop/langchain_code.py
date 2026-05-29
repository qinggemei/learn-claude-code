#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.

Usage:
    pip install anthropic python-dotenv
    ANTHROPIC_API_KEY=... python s01_agent_loop/code.py
"""
import os
import subprocess

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

tools = [bash]

tool_dict = {tool.name: tool for tool in tools}

model_with_tools = model.bind_tools(tools)

def agent_loop(messages: list):
    while True:
        # 返回的是工具名与参数 / 返回消息
        # AiMessage
        response = model_with_tools.invoke(messages)

        messages.append(response)

        if (len(response.tool_calls) == 0):
            return

        for tool_call in response.tool_calls:
            # 打印 入参
            print(f"\033[33m$ {tool_call["args"]['command']}\033[0m")
            tool = tool_dict[tool_call["name"]]
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

