import os
import math
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class Settings:
    endpoint: str = 'https://inference.do-ai.run/v1'
    api_key: str = ''
    router_key: str = ''
    planner: str = 'rules'
    planner_model: str = 'hermes-3-llama-3.1-8b'
    models: dict = field(default_factory=lambda: {
        'eco': 'llama-3.2-3b-instruct',
        'balanced': 'qwen2.5-coder-7b-instruct',
        'heavy': 'llama3.3-70b-instruct',
    })
    wh_per_1k: dict = field(default_factory=lambda: {'eco': .05, 'balanced': .15, 'heavy': .5})
    usd_per_million: dict = field(default_factory=lambda: {'eco': .10, 'balanced': .30, 'heavy': 1.00})
    cache_size: int = 256
    cache_ttl: float = 900
    timeout: float = 90
    demo_fail_eco: bool = False

    @property
    def mock(self):
        return not self.api_key

    @classmethod
    def from_env(cls):
        load_dotenv()
        s = cls()
        s.endpoint = os.getenv('DO_INFERENCE_ENDPOINT', s.endpoint).rstrip('/')
        s.api_key = os.getenv('DO_API_KEY', '').strip()
        s.router_key = os.getenv('ROUTER_API_KEY', '')
        s.planner = os.getenv('PLANNER_MODE', 'rules')
        if s.planner not in {'rules', 'model'}:
            raise ValueError('PLANNER_MODE must be rules or model')
        s.planner_model = os.getenv('DO_PLANNER_MODEL', s.planner_model)
        for tier in s.models:
            s.models[tier] = os.getenv(f'DO_MODEL_{tier.upper()}', s.models[tier])
            s.wh_per_1k[tier] = float(os.getenv(f'{tier.upper()}_WH_PER_1K', s.wh_per_1k[tier]))
            s.usd_per_million[tier] = float(os.getenv(f'{tier.upper()}_USD_PER_MILLION', s.usd_per_million[tier]))
        s.cache_size = int(os.getenv('CACHE_MAX_ENTRIES', '256'))
        s.cache_ttl = float(os.getenv('CACHE_TTL_SECONDS', '900'))
        s.timeout = float(os.getenv('UPSTREAM_TIMEOUT_SECONDS', '90'))
        s.demo_fail_eco = os.getenv('DEMO_FAIL_ECO', 'false').lower() == 'true'
        values = [*s.wh_per_1k.values(), *s.usd_per_million.values(), s.cache_size, s.cache_ttl, s.timeout]
        if not all(math.isfinite(v) for v in values) or min(values) < 0 or s.timeout <= 0:
            raise ValueError('Rates/cache bounds must be nonnegative; timeout must be positive.')
        return s
