from .models import Conversation, Message


class ConversationStore:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._conversations: dict[str, Conversation] = {}

    def get_or_create(self, conversation_id: str) -> Conversation:
        if not conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        return self._conversations.setdefault(
            conversation_id, Conversation(conversation_id)
        )

    def add_message(self, conversation_id: str, role: str, content: str) -> Message:
        if not content.strip():
            raise ValueError("content must not be empty")
        c = self.get_or_create(conversation_id)
        message = Message(role, content)
        c.messages.append(message)
        del c.messages[: -self.max_messages]
        return message

    def get_messages(self, conversation_id: str) -> list[Message]:
        return list(self.get_or_create(conversation_id).messages)

    def clear(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)
