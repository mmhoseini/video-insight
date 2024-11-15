from dataclasses import dataclass
from os import getenv
import os
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    host: str = os.environ.get("DB_HOST")
    port: int = int(os.environ.get("DB_PORT"))
    database: str = os.environ.get("DB_NAME")
    user: str = os.environ.get("DB_USER")
    password: str = os.environ.get("DB_PASSWORD")
