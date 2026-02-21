import logging

from rag_core.config.settings import settings


class ExtraAppendingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)

        # Standard LogRecord Keys, die wir NICHT als extra anhängen wollen
        standard = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }

        extras = {k: v for k, v in record.__dict__.items() if k not in standard}
        if extras:
            extra_str = " ".join(f"{k}={extras[k]!r}" for k in sorted(extras))
            return f"{base} | {extra_str}"
        return base


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        ExtraAppendingFormatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )

    root = (
        logging.getLogger()
    )  
    root.handlers.clear()  
    root.addHandler(handler)
    root.setLevel(settings.LOG_LEVEL)
