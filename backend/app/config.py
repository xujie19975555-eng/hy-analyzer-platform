from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "HY Analyzer Platform"
    debug: bool = False

    # API URLs
    hyperliquid_api_url: str = "https://api.hyperliquid.xyz"
    superx_api_url: str = "https://api.superx-webapi.trysuper.co"

    # Rate limiting - Hyperliquid has 100 req/min limit
    api_rate_limit_delay: float = 0.6
    max_records_per_request: int = 10000

    # Claude API (via relay)
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    claude_model: str = "claude-sonnet-4-5-20250929"

    # Second model (Claude Haiku for faster/cheaper second opinion)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"

    # Alternative: Use Claude Haiku as second model
    claude_haiku_model: str = "claude-haiku-4-5-20251001"
    use_dual_claude: bool = True  # If true, use Sonnet + Haiku instead of Claude + OpenAI

    # AI scoring settings
    ai_scoring_timeout: float = 60.0
    ai_scoring_enabled: bool = True

    # AWS S3 settings (for Hyperliquid historical data beyond 10K limit)
    # The bucket uses requester-pays, so you pay ~$0.09/GB transfer
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    # Cache settings
    cache_valid_hours: float = 1.0  # Analysis cache validity (hours)

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
