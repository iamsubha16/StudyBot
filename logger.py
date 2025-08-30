import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Define log file path
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Configure logger
logger = logging.getLogger("GraphRAGLogger")
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)

# Set log format
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)
