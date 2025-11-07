import logging

logger = logging.getLogger('agent_logger')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('logs/agent.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_message(message: str):
    logger.info(message)
