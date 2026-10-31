"""Configuration loaded from environment variables only. No secrets are hardcoded."""
from __future__ import annotations
import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: str = "./data"
    db_path: str = "./data/paper2product.db"
    llm_provider: str = "auto"          # auto | gemini | demo
    llm_model: str = "gemini-2.5-flash"
    llm_api_key: str = ""
    llm_timeout_s: int = 60
    llm_max_retries: int = 2
    llm_price_per_1k_tokens: float = 0.0  # optional, for cost estimates
    sandbox_mode: str = "auto"          # auto | docker | local
    sandbox_image: str = "python:3.12-slim"
    github_token: str = ""
    hf_token: str = ""
    max_upload_mb: int = 20
    max_pdf_pages: int = 60
    api_key: str = ""                   # optional shared key protecting the API
    rate_limit_per_min: int = 20

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        data_dir = e("P2P_DATA_DIR", "./data")
        return cls(
            data_dir=data_dir,
            db_path=e("P2P_DB_PATH", os.path.join(data_dir, "paper2product.db")),
            llm_provider=e("P2P_LLM_PROVIDER", "auto").lower(),
            llm_model=e("P2P_LLM_MODEL", "gemini-2.5-flash"),
            llm_api_key=e("GEMINI_API_KEY", e("GOOGLE_API_KEY", "")),
            llm_timeout_s=_int("P2P_LLM_TIMEOUT_S", 60),
            llm_max_retries=_int("P2P_LLM_MAX_RETRIES", 2),
            llm_price_per_1k_tokens=float(e("P2P_LLM_PRICE_PER_1K", "0") or 0),
            sandbox_mode=e("P2P_SANDBOX", "auto").lower(),
            sandbox_image=e("P2P_SANDBOX_IMAGE", "python:3.12-slim"),
            github_token=e("GITHUB_TOKEN", ""),
            hf_token=e("HF_TOKEN", ""),
            max_upload_mb=_int("P2P_MAX_UPLOAD_MB", 20),
            max_pdf_pages=_int("P2P_MAX_PDF_PAGES", 60),
            api_key=e("P2P_API_KEY", ""),
            rate_limit_per_min=_int("P2P_RATE_LIMIT_PER_MIN", 20),
        )
