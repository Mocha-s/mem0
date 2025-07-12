import os
from uuid import UUID

USER_ID = os.getenv("USER", "default_user")
DEFAULT_APP_ID = UUID("550e8400-e29b-41d4-a716-446655440000")  # 固定的UUID