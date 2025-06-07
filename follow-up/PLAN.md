# Intelligent Conversation Follow-Up System

An AI-powered system that analyzes user conversations and generates personalized conversation starters to re-engage users.

## Overview

This system uses two specialized AI agents to:
1. **Analyze conversations** - Segment and rate user engagement/enjoyment (Agent 1: o3)
2. **Generate follow-ups** - Create compelling conversation starters (Agent 2: o3 + tool_deep_research)

## Project Status

### ✅ Step 1: Database Integration (DONE)
- **Status**: Complete
- **Code**: `chat-retrieval.py`
- **Description**: Database connectivity and chat access functionality implemented

### 🔄 Step 2: Conversation Analyzer Agent (Agent 1)
- **Model**: o3
- **Input**: Single chat conversation
- **Task**: Segment and analyze conversation using this prompt:

```
Analyze this dialogue and create a structured breakdown:
1. Identify key thematic segments
2. For each segment, describe: topic, tone, conversation direction
3. Rate user engagement (1-10) and enjoyment (1-10) with brief justifications
4. Output structured JSON using Pydantic
```

- **Processing**: Sort segments by combined engagement + enjoyment ratings (highest first)
- **Output**: Structured JSON with conversation segments and ratings

### 🔄 Step 3: Conversation Starter Generator Agent (Agent 2)
- **Model**: o3 + tool_deep_research
- **Input**: Top 3 highest-rated conversation  (ranked by the sum of enjoyment and engagement scores) from Agent 1
- **Task**: Generate re-engagement notifications using this enhanced prompt:

```
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

OUTPUT: Generate 30 ranked conversation starters (1-30, best to worst)
```

### Output Structure
```python
class ConversationStarter(BaseModel):
    id: int  # Quality rank (1 = best)
    context: str  # Relevant situation/segment
    starter: str  # The conversation starter text

class ConversationStarterList(BaseModel):
    starters: list[ConversationStarter]
```

## Workflow Architecture

### Environment Setup
- **Project Location**: `C:\Users\nursu\OneDrive\Desktop\Atarino REPOs\follow-up`
- **Self-Sufficient**: Must be runnable without referencing other projects
- **Reference Code**: Use patterns from bachmanity project and atarino-task-manager\view_conversations.py

### Processing Flow
1. **Database Access** → Read user conversations (✅ Done)
2. **Agent 1** → Segment and rate conversations → JSON output
3. **Agent 2** → Generate 30 ranked conversation starters → JSON output
4. **Terminal Output** → Display ranked starters in readable format

## Deliverables

1. **Functional workflow** with both agents integrated
2. **Clean terminal output** displaying ranked conversation starters in readable JSON format
3. **requirements.txt** with all necessary dependencies
4. **README.md** with setup instructions and usage examples

## Success Criteria
- System processes conversations end-to-end
- Generates contextually relevant, engaging conversation starters
- Code is minimal, clean, and immediately functional
- Output is properly formatted and ranked

**Focus**: Build the simplest possible solution that delivers the core functionality. Avoid over-engineering.



