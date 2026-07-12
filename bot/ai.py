"""Клиент Timeweb Cloud AI: OpenAI-совместимый API агента.

Документация: https://timeweb.cloud/docs/ai-agents/api-usage/openai-compatible-api
Модель (GPT-5.4) выбирается в настройках агента в панели Timeweb Cloud;
поле model в запросе отправляется только для совместимости.
"""

import logging

import aiohttp

log = logging.getLogger("ai")


class AiClient:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        agent_id: str,
        token: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ):
        self._session = session
        self._url = f"{base_url}/api/v1/cloud-ai/agents/{agent_id}/v1/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def chat(self, messages: list[dict]) -> str | None:
        body = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        log.info("Timeweb AI запрос: url=%s body=%s", self._url, body)
        try:
            async with self._session.post(
                self._url,
                json=body,
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    log.error(
                        "Timeweb AI ответил %s: %s", resp.status, await resp.text()
                    )
                    return None
                payload = await resp.json(content_type=None)
        except aiohttp.ClientError as exc:
            log.error("Ошибка запроса к Timeweb AI: %s", exc)
            return None

        log.info("Timeweb AI ответ: %s", payload)
        try:
            text = payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError):
            log.error("Неожиданный формат ответа Timeweb AI: %s", payload)
            return None
        return text or None
