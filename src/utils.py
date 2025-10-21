import os
from typing import List

giga_api_key = os.getenv("GIGACHAT_API_KEY", "")
giga_api_scope = os.getenv("GIGACHAT_API_SCOPE", "GIGACHAT_API_B2B")