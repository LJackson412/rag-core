import os

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()


class Settings:
    API_KEY: SecretStr = SecretStr(os.environ["OPENAI_API_KEY"])

    POPPLER_PATH = os.getenv("POPPLER_PATH")
    TESSERACT_PATH = os.getenv("TESSERACT_PATH")
    TESSDATA_PATH = os.getenv("TESSDATA_PATH")

    if POPPLER_PATH:
        os.environ["PATH"] += os.pathsep + POPPLER_PATH
    if TESSERACT_PATH:
        os.environ["PATH"] += os.pathsep + TESSERACT_PATH
    if TESSDATA_PATH:
        os.environ["PATH"] += os.pathsep + TESSDATA_PATH


settings = Settings()
