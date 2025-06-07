import os

from dotenv import load_dotenv
from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()  # Load environment variables from .env file

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PWD = os.getenv("POSTGRES_PWD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_DB = os.getenv("POSTGRES_DB")

engine = create_engine(
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PWD}@{POSTGRES_HOST}/{POSTGRES_DB}"
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Dependency to get the database session
def get_db():
    db = SessionLocal()  # Create a new session
    try:
        yield db  # Yield the session to the caller
    finally:
        db.close()  # Close the session when done
