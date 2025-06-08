import logging
import os
import sys
from typing import List, Optional

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from openai_model import get_openai_model

load_dotenv()

# Set up basic logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


class EmailAgentResult(BaseModel):
    greeting: str
    email: str
    farewell: str


class ConversationFollowup(BaseModel):
    followup: str
    category: str
    mood: str
    engagement_score: int
    personalization: str
    context: str
    topic: str
    interest_level: str
    research: str
    psychology: str
    strategy: str


@dataclass
class EmailAgentDeps:
    conversation_followup: ConversationFollowup


def make_email_agent(model_name="o3"):
    agent = Agent(
        get_openai_model(model_name),
        deps_type=EmailAgentDeps,
        retries=3,
        output_type=EmailAgentResult,
    )

    @agent.system_prompt
    def system_prompt(ctx: RunContext[EmailAgentDeps]) -> str:
        return f"""
        You are a very empathetic and friendly email assistant. You had previous conversation with the user and obtained some topic starters and followups based on
        that past conversation.

        Email should always be positive. If user's mood is also positive, give them a reason to be even more positive. If negative - then cheer them up and be inspirational.

        You can add jokes occasionally but don't overdo it and don't be obnoxiously funny and only if the joke serves the purpose of the email, which is make the user to come back to the application.

        Email should be long enough to be interesting and engaging but not more than 1 average paragraph.

        As a farewell - ask user to return to our website `https://atarino.io/app/home` and continue the conversation there.

        Followups are somewhat detailed and contain information below. Use that to create an email text that you will send to the user.
        
        Context Information:
        - Conversation Followup: {ctx.deps.conversation_followup.followup}
        - Current Mood: {ctx.deps.conversation_followup.mood}
        - Additional Context: {ctx.deps.conversation_followup.context}
        - Topic: {ctx.deps.conversation_followup.topic}
        - Interest Level: {ctx.deps.conversation_followup.interest_level}
        - Research: {ctx.deps.conversation_followup.research}
        - Psychology: {ctx.deps.conversation_followup.psychology}
        - Strategy: {ctx.deps.conversation_followup.strategy}

        Most important part of the followup is the `Conversation Followup`. Word the entire email around that.
        """

    return agent
