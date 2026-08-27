"""
BaseAgent: shared LLM interface and utilities for all ResearchForge agents.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Abstract base class for all ResearchForge-ECRM agents.
    Wraps LLM calls with consistent prompt formatting.
    """

    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialise LLM client based on settings."""
        if settings.llm_provider == "openai":
            if not settings.openai_api_key:
                logger.info("[%s] No OpenAI key configured; using deterministic offline mode.", self.name)
                return
            try:
                import openai  # type: ignore
                self._client = openai.OpenAI(api_key=settings.openai_api_key)
                self._model = settings.openai_model
                logger.info("[%s] Using OpenAI (%s).", self.name, self._model)
            except Exception as exc:
                logger.warning("[%s] OpenAI unavailable: %s", self.name, exc)
        elif settings.llm_provider == "anthropic":
            if not settings.anthropic_api_key:
                logger.info("[%s] No Anthropic key configured; using deterministic offline mode.", self.name)
                return
            try:
                import anthropic  # type: ignore
                self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                self._model = settings.anthropic_model
                logger.info("[%s] Using Anthropic (%s).", self.name, self._model)
            except Exception as exc:
                logger.warning("[%s] Anthropic unavailable: %s", self.name, exc)
        else:
            logger.info("[%s] Running in mock LLM mode.", self.name)

    def llm_call(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        """Call the configured LLM and return the text response."""
        if self._client is None:
            return self._mock_llm(system_prompt, user_prompt)

        try:
            if settings.llm_provider == "openai":
                return self._openai_call(system_prompt, user_prompt, json_mode)
            elif settings.llm_provider == "anthropic":
                return self._anthropic_call(system_prompt, user_prompt)
        except Exception as exc:
            logger.error("[%s] LLM call failed: %s – using mock.", self.name, exc)
        return self._mock_llm(system_prompt, user_prompt)

    def _openai_call(self, system: str, user: str, json_mode: bool) -> str:
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _anthropic_call(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=settings.llm_max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _mock_llm(self, system: str, user: str) -> str:
        """Deterministic mock for offline testing."""
        if self.name == "HypothesisAgent":
            return json.dumps({
                "hypothesis": "Test a regularized, context-aware model variant and compare it against the current baseline.",
                "rationale": "A bounded intervention permits reproducible evidence and failure attribution.",
            })
        if self.name == "AnalyzerAgent":
            return json.dumps({
                "finding": "The experiment produced a measured validation outcome.",
                "claim": "The intervention is retained only when its outcome is supported by the recorded experiment.",
                "supports_hypothesis": True,
                "failure_flags": ["None"],
            })
        return (
            f"[MOCK RESPONSE from {self.name}]\n"
            f"System: {system[:100]}...\n"
            f"User: {user[:100]}...\n"
            f"Generated: A new hypothesis based on the research context."
        )

    def parse_json_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling markdown code fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("[%s] Failed to parse JSON response.", self.name)
            return {"raw": text}
