from typing import List
from dataclasses import dataclass
from pydantic import BaseModel
import importlib.util
import os
import sys
from pydantic_ai import Agent, RunContext
import json
from firecrawl import FirecrawlApp

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from openai_model import get_openai_model

# Import ConversationSegment from the chat segmenter rater file
spec = importlib.util.spec_from_file_location(
    "chat_segmenter_rater", 
    os.path.join(os.path.dirname(__file__), "chat-segmenter-rater.py")
)
chat_segmenter_rater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat_segmenter_rater)

ConversationSegment = chat_segmenter_rater.ConversationSegment


class ConversationStarter(BaseModel):
    rank: int  # Quality rank (1 = best)
    context: str  # Relevant situation/segment
    starter: str  # The conversation starter text


class ConversationStarterList(BaseModel):
    starters: List[ConversationStarter]


@dataclass
class StarterGeneratorDeps:
    top_segments: List[ConversationSegment]  # Highest rated segments


class StarterGeneratorResult(BaseModel):
    data: List[ConversationStarter]


def make_agent_conversation_starter_generator(model_name="o3"):
    """Creates a conversation starter generator agent using o3 model with deep research capabilities"""
    
    # Get the OpenAI model
    model = get_openai_model(model_name)
    
    # Initialize Firecrawl for research (will use FIRECRAWL_API_KEY from environment)
    try:
        firecrawl = FirecrawlApp()
    except Exception as e:
        print(f"Warning: Firecrawl not initialized. Research capabilities disabled: {e}")
        firecrawl = None
    
    # Create the enhanced prompt for conversation starter generation
    system_prompt = """
Create conversation starters that compel users to reopen the app. Each starter must be:

CONSTRAINTS:
- Under 15 words
- Proactive (add new value, don't just follow up)
- Show genuine care for the user

VALUE CATEGORIES (choose one):
- 🎯 Useful: reminders, insights, professional advice
- 😄 Funny: humor that makes them smile
- 🤔 Intriguing: curiosity-sparking content
- 💡 Insightful: fresh perspectives on past topics

PROVEN TEMPLATES:
- [Past topic] + [New info/statistics]
- [User interest] + [Timely recommendation]
- [Personal compliment] + [Actionable tip]
- [Previous location] + [Relevant discovery]
- [Recent discussion] + [Fresh angle]
- [Their emotion] + [Encouraging follow-up]
- [Problem mentioned] + [Solution found]
- [Shared moment] + [Playful callback]
- [Unexpected twist] + [Intriguing detail]
- [Familiar concept] + [Surprising fact]

You have access to a deep_research tool that can search the web for current information, statistics, and fresh perspectives on any topic. Use this tool to enhance conversation starters with:
- Recent news or developments related to conversation topics
- Current statistics or data points
- New discoveries or insights
- Trending information that would intrigue the user

You will be given top conversation segments with engagement and enjoyment scores. Use these along with research findings to generate contextually relevant conversation starters.

OUTPUT: Generate 30 ranked conversation starters (1-30, best to worst). Return as JSON with the ConversationStarterList structure.
"""

    agent = Agent(
        model=model,
        deps_type=StarterGeneratorDeps,
        system_prompt=system_prompt,
        result_type=ConversationStarterList
    )
    
    @agent.tool
    async def deep_research(ctx: RunContext[StarterGeneratorDeps], query: str) -> str:
        """
        Research current information, statistics, and insights on a given topic.
        
        Args:
            ctx: The run context containing dependencies
            query: The research query or topic to investigate
            
        Returns:
            Research findings including current information, statistics, and insights
        """
        if not firecrawl:
            return f"Research unavailable for '{query}' - Firecrawl not configured"
        
        try:
            # Search for current information on the topic
            search_results = firecrawl.search(
                query=query,
                limit=3,  # Get top 3 results
                search_format="markdown"
            )
            
            research_summary = f"Research findings for '{query}':\n\n"
            
            if search_results and 'data' in search_results:
                for i, result in enumerate(search_results['data'][:3], 1):
                    title = result.get('title', 'No title')
                    content = result.get('markdown', result.get('content', ''))[:300]  # First 300 chars
                    url = result.get('url', 'No URL')
                    
                    research_summary += f"{i}. {title}\n"
                    research_summary += f"   Content: {content}...\n"
                    research_summary += f"   Source: {url}\n\n"
            else:
                research_summary += "No recent information found for this topic."
                
            return research_summary
            
        except Exception as e:
            return f"Research error for '{query}': {str(e)}"
    
    async def run(deps: StarterGeneratorDeps) -> StarterGeneratorResult:
        """
        Generate conversation starters from top segments using o3 model with deep research
        """
        
        # Prepare the input context for the agent
        segments_context = ""
        research_topics = []
        
        for i, segment in enumerate(deps.top_segments):
            segments_context += f"""
Segment {i+1} (Combined Score: {segment.combined_score}/20):
- Topic: {segment.topic}
- Tone: {segment.tone}
- Direction: {segment.conversation_direction}
- Engagement: {segment.engagement_score}/10 ({segment.engagement_justification})
- Enjoyment: {segment.enjoyment_score}/10 ({segment.enjoyment_justification})
- Content Preview: {segment.content[:200]}...

"""
            # Extract key topics for research
            research_topics.append(segment.topic)
        
        # Create research-enhanced prompt
        prompt = f"""
Based on these top conversation segments from a user's chat history, generate 30 compelling conversation starters ranked from best (1) to worst (30):

{segments_context}

RESEARCH STRATEGY:
1. Use the deep_research tool to find current information, statistics, or interesting developments related to these topics: {', '.join(research_topics[:5])}
2. Look for recent news, surprising facts, or fresh perspectives that could make compelling conversation starters
3. Focus on information that would genuinely interest someone who previously engaged with these topics
4. Consider the variety of topics and engagement patterns shown across these segments

Generate conversation starters that would make this user want to reopen the app and continue engaging. Incorporate research findings to add new value and intrigue. Focus on the topics, emotions, and interests shown in these high-scoring segments.
"""
        
        result = await agent.run(prompt, deps=deps)
        
        # Extract the starters from the structured result
        starters = result.data.starters
        
        return StarterGeneratorResult(data=starters)
    
    # Return object with run method
    class ConversationStarterAgent:
        def __init__(self):
            self.run = run
    
    return ConversationStarterAgent() 