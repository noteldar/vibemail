from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    TIMESTAMP,
    BigInteger,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class UserConversation(Base):
    __tablename__ = "user_conversations"

    conversation_id = Column(BigInteger, primary_key=True, index=True)  # bigint NOT NULL
    conversation_start = Column(TIMESTAMP(timezone=True), nullable=True)  # timestamp with time zone
    conversation_end = Column(TIMESTAMP(timezone=True), nullable=True)  # timestamp with time zone
    conversation = Column(String, nullable=True)  # character varying
    user_id = Column(String, index=True, nullable=True)  # character varying
    is_agent = Column(Boolean, nullable=True)  # boolean
    word_count = Column(Integer, nullable=True)  # integer
    character_count = Column(Integer, nullable=True)  # integer
    agent_name = Column(Text, nullable=True)  # text
    agent_metadata = Column(JSON, nullable=True)  # jsonb

    __table_args__ = (
        ForeignKeyConstraint(
            ["user_id"], ["users.user_id"], name="user_conversations_users_fk"
        ),
    )

    # Additional validation can be added using Pydantic
    class Config:
        orm_mode = True  # Allows compatibility with ORM 