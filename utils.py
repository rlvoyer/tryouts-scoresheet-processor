import logging

from colorlog import ColoredFormatter

formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    }
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.handlers = []
logger.addHandler(handler)
