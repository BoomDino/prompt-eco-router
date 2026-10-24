"""Conservative keyword fingerprint with an exact request guard; process-local LRU."""
import copy
import hashlib
import json
import re
import time
from collections import OrderedDict


class SemanticCache:
    def __init__(self, size=256, ttl=900):
        self.size, self.ttl = size, ttl
        self.entries = OrderedDict()

    @staticmethod
    def key(request):
        data = request.model_dump(exclude_none=True)
        for field in ('stream', 'stream_options', 'cache'):
            data.pop(field, None)
        exact = json.dumps(data, sort_keys=True, ensure_ascii=False)
        keywords = sorted(set(re.findall(r'\w+', exact.lower())))
        return hashlib.sha256((' '.join(keywords) + '\0' + exact).encode()).hexdigest()

    def get(self, key):
        value = self.entries.get(key)
        if value is None:
            return None
        expires, result = value
        if expires <= time.monotonic():
            del self.entries[key]
            return None
        self.entries.move_to_end(key)
        return copy.deepcopy(result)

    def put(self, key, result):
        if self.size == 0:
            return
        self.entries[key] = (time.monotonic() + self.ttl, copy.deepcopy(result))
        self.entries.move_to_end(key)
        while len(self.entries) > self.size:
            self.entries.popitem(last=False)
