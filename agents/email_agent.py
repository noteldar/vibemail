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


class EmailGenerationAgentResult(BaseModel):
    email_html: str


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


@dataclass
class EmailGenerationAgentDeps:
    email_content: EmailAgentResult
    mood: str


def make_email_content_agent(model_name="o3-mini"):
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


def make_email_generation_agent(model_name="o3-mini"):
    agent = Agent(
        get_openai_model(model_name),
        deps_type=EmailGenerationAgentDeps,
        retries=3,
        output_type=EmailGenerationAgentResult,
    )

    @agent.system_prompt
    def system_prompt(ctx: RunContext[EmailGenerationAgentDeps]) -> str:
        return f"""
        You are a greatest web-designer who's trying to apply his skills and expertise in creating greatest looking visually appealing email.
        You are smart so you know that not everything that work in htmls works in emails, like javascript. So your email html is self-contained and doesn't rely on external resources
        and doesn't have scripts. Just pure html+inline css and good old creativity and your amazing design skills.

        In addition to email textual content, like greeting, email body and farewell messages, there's the overall mood available to you. Use the mood to adjust your final email template
        like using appropriate colors, fonts, etc.

        Experiment with different fonts, colors, and layouts. Follow this thought process:
        Based on the overall mood of the email, decide on the overall style and vibe that you will follow.
        Examples of styles and vibe can be:
        1) Tech-minimalistic, California-lounge, Texas-patriotic, Rustic-starwars meets-charley chaplin.
        2) Suggest colors, fonts, layouts, interactive elements 
        3) Generate the HTML and self check if it will be rendered correctly in the email.

        Between message body and farewell there will be an image. For now just add placeholder with the following html tag:
        <img src="data:image/png;base64,[BASE64_DATA]" alt="Image Description">


        Here's the textual content:
        - Greeting: {ctx.deps.email_content.greeting}
        - Email Body: {ctx.deps.email_content.email}
        - Farewell: {ctx.deps.email_content.farewell}
        - Mood: {ctx.deps.mood}

        """

    return agent
