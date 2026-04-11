import json
import logging
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    twilio_whatsapp_from: str = ""

    tavily_api_key: str = ""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"

    family_md_path: str = "./family.md"
    whisper_model_size: str = "large-v3"  # used by whisper container only
    whisper_url: str = "http://localhost:8080"
    triton_url: str = "localhost:8001"

    # Stored as a JSON string in .env, e.g. '{"+447911123456": "Alice"}'
    phone_to_name: str = "{}"

    model_config = {"env_file": ".env"}

    def get_caller_name(self, phone: str) -> str | None:
        try:
            mapping = json.loads(self.phone_to_name)
        except json.JSONDecodeError:
            logger.error("PHONE_TO_NAME env var is not valid JSON: %r", self.phone_to_name)
            return None
        return mapping.get(phone)


settings = Settings()
