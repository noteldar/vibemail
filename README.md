# Intelligent Conversation Follow-Up System

An AI-powered system that analyzes user conversations and generates personalized conversation starters to re-engage users.

## Overview

This system uses two specialized AI agents to:
1. **Analyze conversations** - Segment and rate user engagement/enjoyment (Agent 1: o3)
2. **Generate follow-ups** - Create compelling conversation starters (Agent 2: o3 + tool_deep_research)

## Features

- 🔍 **Conversation Analysis**: Automatically segments conversations and rates engagement
- 🎯 **Smart Follow-ups**: Generates 30 ranked conversation starters
- 📊 **Engagement Scoring**: Rates user engagement and enjoyment (1-10 scale)
- 🔄 **End-to-End Workflow**: Seamless processing from conversation to follow-up
- 💾 **Database Integration**: SQLite-based conversation storage

## Quick Start

### 1. Setup Environment

```bash
# Clone or create the project directory
cd follow-up

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here  # Optional for research features
```

### 3. Run the System

```bash
python workflow.py
```

## System Architecture

### Agent 1: Conversation Analyzer (o3)
- **Input**: Single chat conversation
- **Task**: Segment and analyze conversation
- **Output**: Structured JSON with engagement/enjoyment ratings

### Agent 2: Conversation Starter Generator (o3 + tool_deep_research)
- **Input**: Top 3 highest-rated conversation segments
- **Task**: Generate 30 ranked conversation starters
- **Output**: Ranked list of compelling follow-ups

## Usage Examples

### Process User Conversations
```python
from workflow import FollowUpWorkflow

workflow = FollowUpWorkflow()

# Process all conversations for a user
starters = await workflow.process_user_conversations("user_123")

# Process a single conversation
starters = await workflow.process_single_conversation("conv_1")
```

### Sample Output
```json
{
  "user_id": "user_123",
  "total_starters": 30,
  "starters": [
    {
      "rank": 1,
      "context": "User showed high engagement discussing strength training",
      "starter": "Found a new study: compound movements boost metabolism 24hrs post-workout! 💪"
    },
    {
      "rank": 2,
      "context": "User excited about Japan travel planning",
      "starter": "Cherry blossom forecast just updated - perfect timing for your Tokyo trip! 🌸"
    }
  ]
}
```

## Project Structure

```
follow-up/
├── models.py                    # Pydantic data models
├── openai_model.py             # OpenAI model configuration
├── conversation_analyzer.py     # Agent 1: Conversation analysis
├── conversation_starter_generator.py  # Agent 2: Starter generation
├── database.py                 # Mock database with sample data
├── firecrawl_tools.py          # Research tools integration
├── workflow.py                 # Main orchestration workflow
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Conversation Starter Categories

The system generates starters across 4 value categories:

- 🎯 **Useful**: Reminders, insights, professional advice
- 😄 **Funny**: Humor that makes users smile
- 🤔 **Intriguing**: Curiosity-sparking content
- 💡 **Insightful**: Fresh perspectives on past topics

## Sample Data

The system includes sample conversations covering:
- **Fitness**: Strength training discussion
- **Productivity**: Deep Work book discussion
- **Travel**: Japan trip planning

## API Requirements

- **OpenAI API**: Required for both agents (o3 model)
- **Firecrawl API**: Optional for enhanced research capabilities

## Customization

### Adding New Conversation Sources
Modify `database.py` to connect to your actual conversation database:

```python
def get_user_conversations(self, user_id: str) -> List[ChatConversation]:
    # Replace with your database query logic
    pass
```

### Adjusting Analysis Criteria
Modify the system prompts in `conversation_analyzer.py` and `conversation_starter_generator.py` to customize analysis and generation behavior.

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure `.env` file contains valid OpenAI API key
2. **No Conversations Found**: Check user_id exists in sample data
3. **Empty Starters**: Verify conversations have sufficient content for analysis

### Logging

The system provides detailed logging. Check console output for:
- Conversation analysis progress
- Segment extraction results
- Starter generation status

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For questions or issues:
- Check the logs for detailed error information
- Ensure all dependencies are properly installed
- Verify API keys are correctly configured
