import os

DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_NAME = os.getenv("DB_NAME", "ML_FINANCE")

MODEL_FOLDER = "models/"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS = 8192

# App
APP_LANGUAGE = "ro"
APP_CURRENCY_SYMBOL = "€"
APP_NUMBER_UNIT = 1_000  # display in thousands