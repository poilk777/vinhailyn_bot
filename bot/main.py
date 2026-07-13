"""Логика бота: слушает беседы, решает, когда ответить, и ходит в Timeweb AI."""

import asyncio
import json
import logging
import random
import re
from collections import deque
from pathlib import Path

import aiohttp

from .ai import AiClient
from .config import FEW_SHOT_EXAMPLES, Config
from .vk import VkApiError, VkClient

log = logging.getLogger("bot")

# peer_id бесед начинается с 2 000 000 000
CHAT_PEER_OFFSET = 2_000_000_000

# Сколько сообщений подряд можно отправить как «пузыри» одного ответа
MAX_BUBBLES = 3

_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+)\*(?!\*)")
_MD_BULLET_RE = re.compile(r"^[ \t]*[-*•][ \t]+", re.MULTILINE)
_MD_HEADER_RE = re.compile(r"^#{1,6}[ \t]+", re.MULTILINE)


def _strip_markdown(text: str) -> str:
    # Страховка на случай, если модель всё же вставит форматирование,
    # несмотря на запрет в системном промпте — в VK-чате оно смотрится чужеродно.
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_BULLET_RE.sub("", text)
    text = _MD_HEADER_RE.sub("", text)
    return text


def split_into_bubbles(text: str) -> list[str]:
    """Разбивает ответ на отдельные сообщения по двойному переносу строки —
    так модель может имитировать человека, печатающего несколько сообщений подряд."""
    text = _strip_markdown(text).strip()
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not parts:
        return [text] if text else []
    return parts[:MAX_BUBBLES]


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
        # Таймеры «написать первым», если долго никто не пишет
        self._idle_tasks: dict[int, asyncio.Task] = {}
        self._load_history()

    def _peer_history(self, peer_id: int) -> deque:
        if peer_id not in self._history:
            self._history[peer_id] = deque(maxlen=self.config.history_size)
        return self._history[peer_id]

    def _peer_lock(self, peer_id: int) -> asyncio.Lock:
        if peer_id not in self._locks:
            self._locks[peer_id] = asyncio.Lock()
        return self._locks[peer_id]

    def _load_history(self) -> None:
        if not self.config.history_file:
            return
        path = Path(self.config.history_file)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for peer_id, items in data.items():
                self._history[int(peer_id)] = deque(items, maxlen=self.config.history_size)
            log.info("История загружена из %s (%d бесед)", path, len(self._history))
        except Exception:
            log.exception("Не удалось загрузить историю из %s", path)

    def _save_history(self) -> None:
        if not self.config.history_file:
            return
        path = Path(self.config.history_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {str(peer_id): list(d) for peer_id, d in self._history.items()}
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            log.exception("Не удалось сохранить историю в %s", path)

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

    async def _generate_and_send(
        self, peer_id: int, history: deque, nudge: str | None = None
    ) -> None:
        """Просит ИИ сгенерировать ответ по истории беседы и отправляет его.
        nudge — служебная подсказка модели (не сохраняется в историю), используется,
        когда бот пишет сам без повода (см. _send_idle_nudge)."""
        async with self._peer_lock(peer_id):
            await self.vk.set_typing(peer_id)
            messages = [{"role": "system", "content": self.config.system_prompt}]
            messages.extend(FEW_SHOT_EXAMPLES)
            messages.extend(history)
            if nudge:
                messages.append({"role": "user", "content": nudge})
            answer = await self.ai.chat(messages)
            if not answer:
                return
            history.append({"role": "assistant", "content": answer})
            self._save_history()
            bubbles = split_into_bubbles(answer)
            for bubble in bubbles:
                await self.vk.set_typing(peer_id)
                # Пауза перед отправкой, примерно как время печати человеком
                delay = min(0.5 + len(bubble) / 25, 4.0) * random.uniform(0.7, 1.3)
                await asyncio.sleep(delay)
                await self.vk.send_message(peer_id, bubble)
            log.info(
                "Ответил в peer %s (%s сообщение(й), %s символов)%s",
                peer_id, len(bubbles), len(answer),
                " [сам начал разговор]" if nudge else "",
            )

    def _schedule_idle_nudge(self, peer_id: int) -> None:
        if self.config.idle_nudge_minutes <= 0:
            return
        old = self._idle_tasks.get(peer_id)
        if old and not old.done():
            old.cancel()
        self._idle_tasks[peer_id] = asyncio.create_task(self._idle_nudge_timer(peer_id))

    async def _idle_nudge_timer(self, peer_id: int) -> None:
        try:
            await asyncio.sleep(self.config.idle_nudge_minutes * 60)
        except asyncio.CancelledError:
            return
        try:
            await self._send_idle_nudge(peer_id)
        except Exception:
            log.exception("Ошибка при попытке написать первым в peer %s", peer_id)

    async def _send_idle_nudge(self, peer_id: int) -> None:
        history = self._peer_history(peer_id)
        if not history:
            return  # не о чем писать без контекста
        log.info(
            "Тишина %s мин. в peer %s — пишу первым",
            self.config.idle_nudge_minutes, peer_id,
        )
        await self._generate_and_send(
            peer_id, history,
            nudge=(
                "[Сюда никто не писал уже долгое время. Напиши сам что-нибудь "
                "короткое, чтобы оживить разговор — вспомни последнюю тему, "
                "спроси что-то новое или пошути. Не упоминай, что было молчание.]"
            ),
        )

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
        self._save_history()

        if is_chat or not self.config.idle_nudge_chats_only:
            self._schedule_idle_nudge(peer_id)

        if not self._should_reply(message, text, is_chat):
            log.info("Решил не отвечать в peer %s (не обращались)", peer_id)
            return

        log.info("Отвечаю в peer %s", peer_id)
        await self._generate_and_send(peer_id, history)

    async def run(self) -> None:
        log.info("Бот запущен, группа %s, слушаю Long Poll…", self.group_id)
        async for update in self.vk.listen(self.group_id):
            log.info("Событие Long Poll: %s", update)
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


async def startup_diagnostics(vk: VkClient, group: dict, debug_chat_local_id: int = 0) -> None:
    """Проверяем от лица VK API, в каких беседах бот состоит и какой у него доступ."""
    log.info("=== ДИАГНОСТИКА ===")
    log.info(
        "Работаю от имени сообщества «%s» (@%s, id %s). "
        "Проверьте, что в беседу добавлено именно ОНО.",
        group.get("name"), group.get("screen_name"), group.get("id"),
    )
    try:
        convs = await vk.call("messages.getConversations", count=20)
    except VkApiError as exc:
        log.error("Не удалось получить список диалогов: %s", exc)
        return

    chat_peers = []
    for item in convs.get("items", []):
        peer = item.get("conversation", {}).get("peer", {})
        log.info("Диалог бота: peer_id=%s type=%s", peer.get("id"), peer.get("type"))
        if peer.get("type") == "chat":
            chat_peers.append(peer["id"])

    if debug_chat_local_id:
        # Принимаем и локальный ID (число из "sel=c123"), и уже готовый peer_id
        # (если пользователь скопировал число >= 2 000 000 000 целиком).
        debug_peer_id = (
            debug_chat_local_id
            if debug_chat_local_id >= CHAT_PEER_OFFSET
            else CHAT_PEER_OFFSET + debug_chat_local_id
        )
        if debug_peer_id not in chat_peers:
            log.info(
                "DEBUG_CHAT_LOCAL_ID=%s задан явно, проверяю peer_id=%s точечно "
                "(даже если его нет в списке диалогов выше)",
                debug_chat_local_id, debug_peer_id,
            )
            chat_peers.append(debug_peer_id)

    if not chat_peers:
        log.warning(
            "Бот НЕ видит ни одной беседы (peer_id вида 2000000xxx). "
            "Значит, либо сообщество реально не состоит в беседе, "
            "либо в беседу добавлено другое сообщество, либо VK ещё не "
            "показал беседу боту (в режиме «только упоминания» беседа "
            "появляется здесь после первого упоминания через @)."
        )

    for peer_id in chat_peers:
        try:
            members = await vk.call("messages.getConversationMembers", peer_id=peer_id)
            log.info(
                "Беседа %s: ✅ есть доступ ко всей переписке (участников: %s)",
                peer_id, members.get("count"),
            )
        except VkApiError as exc:
            if exc.code == 917:
                log.warning(
                    "Беседа %s: ❌ НЕТ доступа ко всей переписке (ошибка 917) — "
                    "режим «только упоминания». Откройте беседу → Управление "
                    "беседой → участники → нажмите на сообщество → включите "
                    "доступ ко всей переписке (или сделайте администратором).",
                    peer_id,
                )
            else:
                log.warning("Беседа %s: не удалось проверить доступ: %s", peer_id, exc)
    log.info("=== КОНЕЦ ДИАГНОСТИКИ ===")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = Config()
    async with aiohttp.ClientSession() as session:
        vk = VkClient(session, config.vk_group_token, config.vk_api_version)
        try:
            group = await vk.get_own_group()
            group_id = config.vk_group_id or group["id"]
        except Exception:
            log.exception(
                "Не удалось определить группу VK — проверьте VK_GROUP_TOKEN "
                "(нужны права «сообщения сообщества») и VK_GROUP_ID"
            )
            raise
        await startup_diagnostics(vk, group, config.debug_chat_local_id)
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
