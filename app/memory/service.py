from collections import defaultdict, deque
from .store import ConversationStore


class MemoryService:
    def __init__(self, max_messages=20):
        self.max_messages = max_messages
        self._m = defaultdict(lambda: deque(maxlen=max_messages))

    def add(self, cid, role, content):
        self._m[cid].append({"role": role, "content": content})

    def get(self, cid):
        return list(self._m[cid])

    def clear(self, cid):
        self._m.pop(cid, None)
