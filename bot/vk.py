"""Минимальный асинхронный клиент VK API + Bots Long Poll."""

import asyncio
import logging
import random
from typing import AsyncIterator

import aiohttp

log = logging.getLogger("vk")

VK_API_URL = "https://api.vk.com/method/"
# Максимальная длина одного сообщения ВКонтакте
VK_MESSAGE_LIMIT = 4096


class VkApiError(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        super().__init__(f"VK API error {code}: {message}")


class VkClient:
    def __init__(self, session: aiohttp.ClientSession, token: str, api_version: str):
        self._session = session
        self._token = token
        self._version = api_version

    async def call(self, method: str, **params) -> dict | list:
        params["access_token"] = self._token
        params["v"] = self._version
        async with self._session.post(VK_API_URL + method, data=params) as resp:
            payload = await resp.json()
        if "error" in payload:
            err = payload["error"]
            raise VkApiError(err.get("error_code", 0), err.get("error_msg", "unknown"))
        return payload["response"]

    async def get_own_group_id(self) -> int:
        response = await self.call("groups.getById")
        groups = response["groups"] if isinstance(response, dict) else response
        return groups[0]["id"]

    async def send_message(self, peer_id: int, text: str) -> None:
        # ВК ограничивает сообщение 4096 символами — длинные ответы режем на части
        for start in range(0, len(text), VK_MESSAGE_LIMIT):
            chunk = text[start : start + VK_MESSAGE_LIMIT]
            await self.call(
                "messages.send",
                peer_id=peer_id,
                message=chunk,
                random_id=random.randint(1, 2**31 - 1),
            )

    async def set_typing(self, peer_id: int) -> None:
        try:
            await self.call("messages.setActivity", peer_id=peer_id, type="typing")
        except VkApiError:
            pass  # индикатор набора не критичен

    async def get_user_names(self, user_ids: list[int]) -> dict[int, str]:
        if not user_ids:
            return {}
        response = await self.call(
            "users.get", user_ids=",".join(map(str, user_ids))
        )
        return {user["id"]: user["first_name"] for user in response}

    async def listen(self, group_id: int) -> AsyncIterator[dict]:
        """Бесконечный цикл Bots Long Poll: отдаёт события message_new и другие."""
        server = key = ts = None
        while True:
            try:
                if server is None:
                    lp = await self.call("groups.getLongPollServer", group_id=group_id)
                    server, key, ts = lp["server"], lp["key"], lp["ts"]
                    log.info("Long Poll сервер получен")
                async with self._session.get(
                    server,
                    params={"act": "a_check", "key": key, "ts": ts, "wait": 25},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    payload = await resp.json(content_type=None)
                if "failed" in payload:
                    if payload["failed"] == 1:
                        ts = payload["ts"]
                    else:
                        server = None  # ключ устарел — переполучаем сервер
                    continue
                ts = payload["ts"]
                for update in payload.get("updates", []):
                    yield update
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                log.warning("Сбой Long Poll (%s), повтор через 5 с", exc)
                server = None
                await asyncio.sleep(5)
