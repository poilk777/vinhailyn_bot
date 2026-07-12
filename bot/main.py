"""Логика бота: слушает беседы, решает, когда ответить, и ходит в Timeweb AI."""

import asyncio
import logging
import random
import re
from collections import deque

import aiohttp

from .ai import AiClient
from .config import Config
from .vk import VkClient

log = logging.getLogger("bot")

# peer_id бесед начинается с 2 000 000 000
CHAT_PEER_OFFSET = 2_000_000_000


class Bot:
    def __init__(self, config: Config, vk: VkClient, ai: AiClient, group_id: int):
        self.config = config
        self.vk = vk
        self.ai = ai
        self.group_id = group_id
        self._mention_re = re.compile(rf"\[club{group_id}\|[^\]]*\]")
        # История сообщений по каждой беседе: [{"role", "content"}, ...]
        self._history: dict[int, deque] = {}
        # Ответы в одной беседе не должны идти параллельно
        self._locks: dict[int, asyncio.Lock] = {}
        self._names: dict[int, str] = {}

    def _peer_history(self, peer_id: int) -> deque:
        if peer_id not in self._history:
            self._history[peer_id] = deque(maxlen=self.config.history_size)
        return self._history[peer_id]

    def _peer_lock(self, peer_id: int) -> asyncio.Lock:
        if peer_id not in self._locks:
            self._locks[peer_id] = asyncio.Lock()
        return self._locks[peer_id]

    async def _sender_name(self, from_id: int) -> str:
        if from_id < 0:
            return "Сообщество"
        if from_id not in self._names:
            names = await self.vk.get_user_names([from_id])
            self._names[from_id] = names.get(from_id, "Участник")
        return self._names[from_id]

    def _is_addressed_to_bot(self, message: dict, text: str) -> bool:
        # Упоминание @сообщества
        if self._mention_re.search(message.get("text", "")):
            return True
        # Ответ на сообщение бота
        reply = message.get("reply_message")
        if reply and reply.get("from_id") == -self.group_id:
            return True
        # Обращение по имени
        lowered = text.lower()
        return any(name in lowered for name in self.config.bot_names)

    def _should_reply(self, message: dict, text: str, is_chat: bool) -> bool:
        if not text:
            return False
        if not is_chat:
            return True  # в личных сообщениях отвечаем всегда
        if self._is_addressed_to_bot(message, text):
            return True
        return random.random() < self.config.random_reply_chance

    async def handle_message(self, message: dict) -> None:
        peer_id = message["peer_id"]
        from_id = message["from_id"]
        log.info(
            "Получено сообщение: peer_id=%s from_id=%s text=%r",
            peer_id, from_id, message.get("text", "")[:80],
        )
        if from_id == -self.group_id:
            return  # своё сообщение

        is_chat = peer_id >= CHAT_PEER_OFFSET

        # Приветствие при добавлении бота в беседу
        action = message.get("action", {})
        if action.get("type") in ("chat_invite_user", "chat_invite_user_by_link"):
            if action.get("member_id") == -self.group_id:
                await self.vk.send_message(
                    peer_id,
                    "Всем привет! Я Винхайлин, буду тут с вами общаться. "
                    "Зовите по имени или отвечайте на мои сообщения 🙂",
                )
            return

        text = self._mention_re.sub("", message.get("text", "")).strip()
        if not text:
            return

        name = await self._sender_name(from_id)
        history = self._peer_history(peer_id)
        history.append({"role": "user", "content": f"{name}: {text}"})

        if not self._should_reply(message, text, is_chat):
            log.info("Решил не отвечать в peer %s (не обращались)", peer_id)
            return

        log.info("Отвечаю в peer %s", peer_id)
        async with self._peer_lock(peer_id):
            await self.vk.set_typing(peer_id)
            messages = [{"role": "system", "content": self.config.system_prompt}]
            messages.extend(history)
            answer = await self.ai.chat(messages)
            if not answer:
                return
            history.append({"role": "assistant", "content": answer})
            await self.vk.send_message(peer_id, answer)
            log.info("Ответил в peer %s (%s символов)", peer_id, len(answer))

    async def run(self) -> None:
        log.info("Бот запущен, группа %s, слушаю Long Poll…", self.group_id)
        async for update in self.vk.listen(self.group_id):
            if update.get("type") != "message_new":
                continue
            message = update.get("object", {}).get("message")
            if not message:
                continue
            # Обрабатываем в фоне, чтобы не блокировать приём событий
            asyncio.create_task(self._safe_handle(message))

    async def _safe_handle(self, message: dict) -> None:
        try:
            await self.handle_message(message)
        except Exception:
            log.exception("Ошибка при обработке сообщения")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = Config()
    async with aiohttp.ClientSession() as session:
        vk = VkClient(session, config.vk_group_token, config.vk_api_version)
        try:
            group_id = config.vk_group_id or await vk.get_own_group_id()
        except Exception:
            log.exception(
                "Не удалось определить группу VK — проверьте VK_GROUP_TOKEN "
                "(нужны права «сообщения сообщества») и VK_GROUP_ID"
            )
            raise
        ai = AiClient(
            session,
            base_url=config.timeweb_base_url,
            agent_id=config.timeweb_agent_id,
            token=config.timeweb_agent_token,
            model=config.ai_model,
            temperature=config.ai_temperature,
            max_tokens=config.ai_max_tokens,
        )
        await Bot(config, vk, ai, group_id).run()


if __name__ == "__main__":
    asyncio.run(main())
