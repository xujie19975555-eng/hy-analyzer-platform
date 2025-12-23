from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "HY Analyzer Platform"
    debug: bool = False

    # API URLs
    hyperliquid_api_url: str = "https://api.hyperliquid.xyz"
    superx_api_url: str = "https://api.superx-webapi.trysuper.co"

    # Rate limiting
    api_rate_limit_delay: float = 0.3
    max_records_per_request: int = 10000

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
