"""Application configuration."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "BTC Signal Dashboard"
    DEBUG: bool = True
    SECRET_KEY: str = "change-this-secret-key"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/btc_signals"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Binance
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET: str = ""
    BINANCE_TIMEFRAME: str = "4h"   # primary timeframe for signal generation

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
