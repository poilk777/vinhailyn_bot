"""Конфигурация бота: всё читается из переменных окружения (.env)."""

import os
from dataclasses import dataclass, field


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(
            f"Не задана обязательная переменная окружения {name}. "
            f"Скопируйте .env.example в .env и заполните значения."
        )
    return value


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        raise SystemExit(f"Переменная {name} должна быть числом, получено: {raw!r}")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        raise SystemExit(f"Переменная {name} должна быть целым числом, получено: {raw!r}")


DEFAULT_SYSTEM_PROMPT = (
    "Ты — Винхайлин, живой и дружелюбный участник беседы ВКонтакте. "
    "Общайся на русском, неформально, с юмором, но без грубости. "
    "Отвечай коротко (1–3 предложения), как обычный человек в чате, "
    "не пиши списки и длинные лекции, не упоминай, что ты ИИ, если не спросят напрямую. "
    "Сообщения участников приходят в формате «Имя: текст» — обращайся к людям по именам."
)


@dataclass
class Config:
    # ВКонтакте
    vk_group_token: str = field(default_factory=lambda: _required("VK_GROUP_TOKEN"))
    vk_group_id: int = field(default_factory=lambda: _int("VK_GROUP_ID", 0))
    vk_api_version: str = field(default_factory=lambda: os.getenv("VK_API_VERSION", "5.199"))

    # Timeweb Cloud AI (OpenAI-совместимый API агента)
    timeweb_agent_id: str = field(default_factory=lambda: _required("TIMEWEB_AGENT_ID"))
    timeweb_agent_token: str = field(default_factory=lambda: _required("TIMEWEB_AGENT_TOKEN"))
    timeweb_base_url: str = field(
        default_factory=lambda: os.getenv(
            "TIMEWEB_BASE_URL", "https://agent.timeweb.cloud"
        ).rstrip("/")
    )
    # Модель реально выбирается в настройках агента в панели Timeweb;
    # это значение отправляется только для совместимости с форматом OpenAI.
    ai_model: str = field(default_factory=lambda: os.getenv("AI_MODEL", "gpt-5.4"))
    ai_temperature: float = field(default_factory=lambda: _float("AI_TEMPERATURE", 1.0))
    ai_max_tokens: int = field(default_factory=lambda: _int("AI_MAX_TOKENS", 600))

    # Поведение бота
    system_prompt: str = field(
        default_factory=lambda: os.getenv("SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT
    )
    # Имена, на которые бот откликается в беседе (через запятую, без учёта регистра)
    bot_names: tuple = field(
        default_factory=lambda: tuple(
            name.strip().lower()
            for name in os.getenv("BOT_NAMES", "винхайлин,виня,vinhailyn").split(",")
            if name.strip()
        )
    )
    # Вероятность (0..1), что бот сам влезет в разговор без обращения к нему
    random_reply_chance: float = field(
        default_factory=lambda: _float("RANDOM_REPLY_CHANCE", 0.05)
    )
    # Сколько последних сообщений беседы держать в контексте для ИИ
    history_size: int = field(default_factory=lambda: _int("HISTORY_SIZE", 30))

    # Временно: локальный ID конкретной беседы (число из адреса vk.com/im?sel=c<число>)
    # для точечной проверки доступа при старте, если беседа не видна через
    # messages.getConversations (типично для уровня «только упоминания»).
    debug_chat_local_id: int = field(
        default_factory=lambda: _int("DEBUG_CHAT_LOCAL_ID", 0)
    )
