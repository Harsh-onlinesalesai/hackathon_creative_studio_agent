import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-2.5-flash")
GENAI_MODEL_LOW = os.getenv("GENAI_MODEL_LOW", "gemini-2.5-flash")
IMAGE_GEN_MODEL = os.getenv("IMAGE_GENAI_MODEL", "gemini-3-pro-image-preview")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from .env file")


# creative upload service config
# Upload Service Configuration
UPLOAD_SERVICE_URL = "https://test.onlinesales.ai/creativeUploadService/v1/upload"
CLIENT_ID = "10065130" # Default Client ID

# Local storage configuration
LOCAL_SAVE_DIR = os.getenv("LOCAL_SAVE_DIR", "./generated_creatives")
SAVE_LOCALLY = os.getenv("SAVE_LOCALLY", "true").lower() == "true"