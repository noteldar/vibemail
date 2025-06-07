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
    starter: str  # The conversation starter text (under 15 words)
    
    # Background & Context
    conversation_context: str  # Detailed background from original conversation
    segment_topic: str  # The specific topic this starter relates to
    user_interest_level: str  # How interested the user seemed in this topic (High/Medium/Low)
    conversation_tone: str  # The tone of the original conversation segment
    
    # Research & Sources
    research_enhanced: bool  # Whether this starter includes internet research
    research_summary: str  # Summary of research findings used (if any)
    sources: List[str]  # List of URLs or sources used for research
    current_relevance: str  # Why this information is timely/current
    
    # Justification & Strategy
    relevance_justification: str  # Why this information is relevant to the user
    engagement_strategy: str  # How this starter will capture user attention
    comeback_psychology: str  # Why this will make the user want to return to the app
    value_category: str  # Which category: Useful, Funny, Intriguing, or Insightful
    
    # Engagement Metrics
    predicted_engagement_score: int  # Predicted engagement score 1-10
    personalization_level: str  # How personalized this is (High/Medium/Low)
    
    # Emotional Tone
    mood: str  # The emotional mood of the conversation starter: Cheerful, Reflective, Gloomy, Humorous, Melancholy, Idyllic, Whimsical, Romantic, Mysterious, Ominous, Calm, Lighthearted, Hopeful, Angry, Fearful, Tense, Lonely


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
Create conversation starters that compel users to reopen the app. Each starter must be under 15 words and include comprehensive background information and justification.

CORE REQUIREMENTS:
- Under 15 words for the starter text
- Proactive (add new value, don't just follow up)
- Show genuine care for the user
- Include detailed context and justification for each starter

VALUE CATEGORIES (choose one for each starter):
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

RESEARCH STRATEGY:
Use the deep_research tool extensively to find current information, statistics, and fresh perspectives. For each research-enhanced starter, you must:
1. Clearly document what research was conducted
2. Provide sources and URLs
3. Explain why the research findings are timely and relevant
4. Connect the research to the user's previous interests

COMPREHENSIVE OUTPUT REQUIREMENTS:
For each conversation starter, provide:

1. BACKGROUND CONTEXT: Detailed explanation of the original conversation segment this relates to
2. RESEARCH DETAILS: If research was used, provide comprehensive summary and sources
3. RELEVANCE JUSTIFICATION: Explain why this information matters to this specific user
4. ENGAGEMENT STRATEGY: Describe how this starter will capture attention
5. COMEBACK PSYCHOLOGY: Explain the psychological reasons why this will make them return
6. PERSONALIZATION ANALYSIS: Assess how personalized this starter is to the user

PSYCHOLOGY OF RETURN ENGAGEMENT:
Consider these psychological triggers that make users return:
- Curiosity gap (partial information that creates desire to know more)
- Personal relevance (connects to their interests/experiences)
- Social proof (others like them found this interesting)
- Time sensitivity (limited time to act or respond)
- Emotional connection (relates to their feelings/experiences)
- Value addition (provides clear benefit or insight)
- Surprise factor (unexpected but relevant information)

OUTPUT: Generate 30 ranked conversation starters (1-30, best to worst). Each must include all required fields with detailed justification and context. Return as JSON with the ConversationStarterList structure.
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
        Research current information, statistics, and insights on a given topic with comprehensive details.
        
        Args:
            ctx: The run context containing dependencies
            query: The research query or topic to investigate
            
        Returns:
            Detailed research findings including current information, statistics, insights, and sources
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
            
            research_summary = f"COMPREHENSIVE RESEARCH REPORT for '{query}':\n\n"
            
            if search_results and 'data' in search_results:
                sources = []
                key_insights = []
                
                for i, result in enumerate(search_results['data'][:3], 1):
                    title = result.get('title', 'No title')
                    content = result.get('markdown', result.get('content', ''))[:1000]  # More content
                    url = result.get('url', 'No URL')
                    
                    sources.append(url)
                    
                    research_summary += f"SOURCE {i}: {title}\n"
                    research_summary += f"URL: {url}\n"
                    research_summary += f"CONTENT PREVIEW: {content}...\n"
                    research_summary += f"RELEVANCE: This source provides current information about {query}\n\n"
                
                research_summary += f"SOURCES LIST: {sources}\n\n"
                research_summary += f"KEY INSIGHTS EXTRACTED:\n"
                research_summary += f"- Current trends and developments in {query}\n"
                research_summary += f"- Recent statistics or data points\n"
                research_summary += f"- Fresh perspectives and expert opinions\n"
                research_summary += f"- Timely information that would interest users\n\n"
                research_summary += f"CONVERSATION STARTER POTENTIAL:\n"
                research_summary += f"- Use surprising facts or statistics from these sources\n"
                research_summary += f"- Reference recent developments to show currency\n"
                research_summary += f"- Connect findings to user's previous interests\n"
                research_summary += f"- Create curiosity gaps with partial information\n"
                
            else:
                research_summary += "No recent information found for this topic.\n"
                research_summary += "FALLBACK STRATEGY: Use general knowledge about this topic with personalized approach based on user's conversation history."
                
            return research_summary
            
        except Exception as e:
            return f"Research error for '{query}': {str(e)}\nFALLBACK: Use conversation history and general knowledge to create engaging starters."
    
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
Based on these top conversation segments from a user's chat history, generate 30 compelling conversation starters ranked from best (1) to worst (30). Each starter must include comprehensive context and justification:

{segments_context}

MANDATORY RESEARCH PHASE:
1. Use the deep_research tool extensively to find current information for these topics: {', '.join(research_topics[:5])}
2. Research additional trending topics that might interest this user
3. Look for surprising statistics, recent developments, or fresh perspectives
4. Document all sources and explain their relevance

COMPREHENSIVE OUTPUT REQUIREMENTS:
For each of the 30 conversation starters, you must provide ALL of these fields:

1. **starter**: The conversation starter text (under 15 words)
2. **conversation_context**: Detailed background from the original conversation segment
3. **segment_topic**: The specific topic this relates to
4. **user_interest_level**: How interested the user seemed (High/Medium/Low)
5. **conversation_tone**: The tone of the original segment
6. **research_enhanced**: Whether internet research was used (true/false)
7. **research_summary**: Comprehensive summary of research findings (if any)
8. **sources**: Array of URLs/sources used for research
9. **current_relevance**: Why this information is timely/current
10. **relevance_justification**: Why this information matters to THIS specific user
11. **engagement_strategy**: How this starter will capture user attention
12. **comeback_psychology**: Psychological reasons why this will make them return
13. **value_category**: "Useful", "Funny", "Intriguing", or "Insightful"
14. **predicted_engagement_score**: Score 1-10 based on personalization and appeal
15. **personalization_level**: "High", "Medium", or "Low"
16. **mood**: The emotional mood of the conversation starter (see MOOD CATEGORIES below)

MOOD CATEGORIES:
Choose the most appropriate emotional mood for each conversation starter:
- **Cheerful**: Upbeat, positive, energetic
- **Reflective**: Thoughtful, contemplative, introspective
- **Gloomy**: Somber, pessimistic, downcast
- **Humorous**: Funny, witty, playful
- **Melancholy**: Wistful, nostalgic, bittersweet
- **Idyllic**: Peaceful, serene, harmonious
- **Whimsical**: Playful, fanciful, quirky
- **Romantic**: Loving, affectionate, intimate
- **Mysterious**: Enigmatic, intriguing, secretive
- **Ominous**: Foreboding, threatening, dark
- **Calm**: Tranquil, composed, relaxed
- **Lighthearted**: Carefree, cheerful, easy-going
- **Hopeful**: Optimistic, encouraging, positive
- **Angry**: Frustrated, irritated, indignant
- **Fearful**: Worried, anxious, concerned
- **Tense**: Stressful, nervous, on-edge
- **Lonely**: Isolated, solitary, yearning for connection

PSYCHOLOGICAL ANALYSIS REQUIRED:
For each starter, analyze:
- What specific psychological trigger will make them want to respond?
- How does this connect to their demonstrated interests?
- What curiosity gap or value proposition will drive re-engagement?
- Why would they prioritize opening the app to respond to this?
- What emotional mood best captures the feeling this starter should evoke?

MOOD SELECTION STRATEGY:
Consider the user's original conversation tone and the desired emotional response:
- Match the user's demonstrated emotional preferences from their conversation history
- Choose moods that complement the value category (e.g., Humorous for funny content, Reflective for deep insights)
- Consider the psychological impact of the mood on user engagement
- Balance variety across different moods to provide emotional diversity

RESEARCH INTEGRATION:
- Use deep_research for at least 15 of the 30 starters
- Include current statistics, recent developments, or trending information
- Always provide sources and explain why the research is relevant
- Connect research findings to the user's conversation history

Generate conversation starters that would make this user want to reopen the app immediately. Focus on creating genuine value, curiosity, and personal connection based on their demonstrated interests and conversation patterns. Each starter must have an appropriate emotional mood that enhances user engagement and matches the content's intended psychological impact.
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