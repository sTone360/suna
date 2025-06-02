# -*- coding: utf-8 -*-
"""Research idea generation pipeline using Suna's architecture.

This module defines a simple multi-stage pipeline orchestrated by
``ThreadManager``.  Each stage represents a specialised agent that
interacts with the LLM and optional tools.  The implementation focuses on
showing how such a pipeline can be built on top of the existing
infrastructure.  Actual scientific reasoning and data retrieval should
be implemented separately.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator, Dict, Any, Optional

from agentpress.thread_manager import ThreadManager
from agentpress.response_processor import ProcessorConfig
from agent.tools.sb_browser_tool import SandboxBrowserTool
from agent.tools.sb_shell_tool import SandboxShellTool
from agent.tools.web_search_tool import SandboxWebSearchTool
from utils.logger import logger


class ResearchIdeaPipeline:
    """Orchestrates multiple specialised agents to generate research ideas."""

    def __init__(self, thread_id: str, project_id: str, *, model_name: str = "anthropic/claude-3-7-sonnet-latest") -> None:
        self.thread_id = thread_id
        self.project_id = project_id
        self.model_name = model_name

        self.thread_manager = ThreadManager()
        self.thread_manager.add_tool(SandboxShellTool, project_id=project_id, thread_manager=self.thread_manager)
        self.thread_manager.add_tool(SandboxBrowserTool, project_id=project_id, thread_manager=self.thread_manager, thread_id=thread_id)
        self.thread_manager.add_tool(SandboxWebSearchTool, project_id=project_id, thread_manager=self.thread_manager)

    async def _run_llm(self, system_prompt: str, user_prompt: str, *, stream: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        """Helper to run a short LLM interaction with the given prompts."""
        system = {"role": "system", "content": system_prompt}
        await self.thread_manager.add_message(self.thread_id, "system", system["content"], is_llm_message=True)
        await self.thread_manager.add_message(self.thread_id, "user", {"content": user_prompt})

        processor = ProcessorConfig(xml_tool_calling=True, execute_tools=True)
        response = await self.thread_manager.run_thread(
            thread_id=self.thread_id,
            system_prompt=system,
            stream=stream,
            llm_model=self.model_name,
            processor_config=processor,
        )

        async for chunk in response:
            yield chunk

    async def stage_one_history(self, topic: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stage 1: gather historic background and recent literature."""
        system_prompt = (
            "You are the 'Scientific History Agent'.  Provide a concise timeline\n"
            "of key milestones and current hotspots related to the topic.  Use\n"
            "the web-search tool when necessary to fetch up-to-date papers."
        )
        user_prompt = f"Study the historic development of {topic} and summarise important publications."
        async for chunk in self._run_llm(system_prompt, user_prompt, stream=True):
            yield chunk

    async def stage_two_problem(self, summary: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stage 2: analyse problems and refine research questions."""
        system_prompt = (
            "You are the 'Problem Analysis Agent'.  Based on the previous summary\n"
            "identify unresolved scientific issues and promising directions."
        )
        user_prompt = f"The following background was collected:\n{summary}\nList concrete research problems that could be explored."
        async for chunk in self._run_llm(system_prompt, user_prompt, stream=True):
            yield chunk

    async def stage_three_reasoning(self, question: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stage 3: deep reasoning using LLM and available tools."""
        system_prompt = (
            "You are the 'Deep Reasoning Agent'.  Conduct multi step reasoning to\n"
            "propose potential solutions or approaches.  Validate calculations\n"
            "using the shell tool when appropriate."
        )
        async for chunk in self._run_llm(system_prompt, question, stream=True):
            yield chunk

    async def run(self, topic: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the pipeline for the given research topic."""
        logger.info("Starting research pipeline")
        collected_summary = ""
        async for chunk in self.stage_one_history(topic):
            collected_summary += json.dumps(chunk)
            yield {"stage": "history", **chunk}

        async for chunk in self.stage_two_problem(collected_summary):
            yield {"stage": "problem", **chunk}
            if chunk.get("type") == "assistant" and "content" in chunk:
                try:
                    content = json.loads(chunk["content"]).get("content", "")
                    question = content
                except Exception:
                    question = chunk["content"]

        if not question:
            question = f"Open questions about {topic}"

        async for chunk in self.stage_three_reasoning(question):
            yield {"stage": "reasoning", **chunk}

