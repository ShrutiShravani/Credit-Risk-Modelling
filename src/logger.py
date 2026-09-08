import logging
import os
from datetime import datetime


# 1. Define filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2. Target the logs folder directory path
logs_dir = os.path.join(os.getcwd(), "logs")

# 3. Create the 'logs' folder ONLY (not the full filename)
os.makedirs(logs_dir, exist_ok=True)

# 4. Construct the complete file path inside the logs directory
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# 5. Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)