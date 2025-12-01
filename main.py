import logging

from config import CONFIG
from experiment import run

logger = logging.getLogger(__name__)


def safe_run(tag, file_suffix: None | str = ""):
    logger.info(f"Starting run: {tag} - {file_suffix}")
    try:
        run(file_suffix)
    except Exception as e:
        logger.exception(f"Run {tag} failed: {e}")


if __name__ == '__main__':
    # First config
    CONFIG.think = False
    CONFIG.gold = True
    safe_run("qwen3-no_think-gs")

    # Second config
    CONFIG.think = False
    CONFIG.gold = False
    safe_run("qwen3-no_think-fp")

    # First config
    CONFIG.think = True
    CONFIG.gold = True
    safe_run("qwen3-think-gs")

    # Second config
    CONFIG.think = True
    CONFIG.gold = False
    safe_run("qwen3-think-fp")
