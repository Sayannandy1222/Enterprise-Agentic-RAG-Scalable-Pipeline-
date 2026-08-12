from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Conversation:
    conversation_id: str
    messages: list[Message] = field(default_factory=list)
