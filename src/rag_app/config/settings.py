import os

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()


class Settings:
    API_KEY: SecretStr = SecretStr(os.environ["OPENAI_API_KEY"])

    POPPLER_PATH = os.getenv("POPPLER_PATH")
    if POPPLER_PATH:
        os.environ["PATH"] += os.pathsep + POPPLER_PATH


settings = Settings()
