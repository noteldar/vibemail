import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
import time

from app.models.UserConversation import UserConversation
from app.db.database import get_db


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ConversationStarter(BaseModel):
    id: int  # Rank of the conversation starter based on quality (1 = highest quality)
    context: str  # Context or situation where this starter is relevant
    starter: str  # Text of the conversation starter sentence

class ConversationStarterList(BaseModel):
    starters: list[ConversationStarter]  # List sorted by quality (best to worst)

# Load environment variables
load_dotenv()

def view_user_conversations(
    user_id="Uojc2mPrSHXzWaGkqRtBsT8xCYb2", 
    days=30, 
    min_conversations=5,
    model="o3",
    temperature=0.9,  # Higher temperature for more creative, human-like starters
    max_retries=3,
    save_to_file=True,
    output_filename=None
):
    """
    Generate conversation starters for a specific user from recent conversations
    
    Args:
        user_id: Target user identifier
        days: How many days back to look for conversations
        min_conversations: Minimum conversations needed to generate starters
        model: OpenAI model to use
        temperature: Model creativity (0.0-1.0)
        max_retries: API retry attempts
        save_to_file: Whether to save output to a text file
        output_filename: Custom filename (auto-generated if None)
    """
    # Setup file output
    output_file = None
    full_output_path = None  # Initialize for later use
    
    if save_to_file:
        # Create the conversation-starters directory if it doesn't exist
        output_dir = "conversation-starters-1"
        os.makedirs(output_dir, exist_ok=True)
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"conversation_starters_{user_id}_{timestamp}.txt"
        
        # Combine directory path with filename
        full_output_path = os.path.join(output_dir, output_filename)
        
        output_file = open(full_output_path, 'w', encoding='utf-8')
        print(f"💾 Output will be saved to: {full_output_path}")
    
    def dual_print(*args, **kwargs):
        """Print to both console and file"""
        print(*args, **kwargs)
        if output_file:
            print(*args, **kwargs, file=output_file)
            output_file.flush()  # Ensure immediate write
    
    # Properly manage database session
    db_generator = get_db()
    db = next(db_generator)
    
    try:
        # Calculate the date from N days ago
        days_ago = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get conversations for the specific user
        conversations = (
            db.query(UserConversation)
            .filter(
                UserConversation.user_id == user_id,
                UserConversation.conversation_start >= days_ago,
            )
            .order_by(UserConversation.conversation_start.desc())
            .all()
        )
        
        if not conversations:
            dual_print(f"\nNo conversations found for user {user_id} in the last {days} days.")
            return

        if len(conversations) < min_conversations:
            dual_print(f"\n⚠️  Only {len(conversations)} conversations found. Need at least {min_conversations} for quality starters.")
            dual_print("Consider increasing the 'days of user activity' parameter or checking if user is active.")
            return

        # Print retrieved conversations for debugging
        dual_print(f"\n📋 Retrieved {len(conversations)} conversations:")
        for conv in conversations[:20]:  # Show first 20 conversations
            speaker = "agent" if conv.is_agent else "user"
            dual_print(f"{speaker}:{conv.conversation}")
        
        if len(conversations) > 20:
            dual_print(f"... and {len(conversations) - 20} more conversations")

        # Format conversations into readable text
        conv_history_formatted = "\n".join([
            f"{'agent' if c.is_agent else 'user'}:{c.conversation}"
            for c in conversations
        ])
        
        # Print formatted conversation history for AI processing
        dual_print(f"\n📝 Formatted conversation history (first 500 chars):")
        dual_print("-" * 50)
        dual_print(conv_history_formatted[:500] + "..." if len(conv_history_formatted) > 500 else conv_history_formatted)
        dual_print("-" * 50)
        
        # Enhanced debug output
        total_words = sum(len(c.conversation.split()) for c in conversations)
        user_messages = sum(1 for c in conversations if not c.is_agent)
        agent_messages = sum(1 for c in conversations if c.is_agent)
        
        dual_print(f"\n📊 Debug Info:")
        dual_print(f"   Total conversations: {len(conversations)}")
        dual_print(f"   User messages: {user_messages}")
        dual_print(f"   Agent messages: {agent_messages}")
        dual_print(f"   Total words: {total_words}")
        dual_print(f"   Average words per message: {total_words // len(conversations) if conversations else 0}")
        
        dual_print(f"\n✅ Chat retrieval completed successfully!")
        dual_print(f"   Found {len(conversations)} conversations for user {user_id}")
        
    except Exception as e:
        dual_print(f"\n❌ Error retrieving conversations: {str(e)}")
        raise
    finally:
        # Close database connection properly
        try:
            next(db_generator)  # This triggers the finally block in get_db()
        except StopIteration:
            pass  # Expected when generator completes
        
        if output_file:
            output_file.close()
            print(f"📁 Output saved to: {full_output_path}")  # Use print instead of dual_print since file is closed

def get_user_conversations_for_workflow(
    user_id="Uojc2mPrSHXzWaGkqRtBsT8xCYb2", 
    days=30, 
    min_conversations=5
):
    """
    Get user conversations for workflow processing (returns data instead of printing)
    
    Args:
        user_id: Target user identifier
        days: How many days back to look for conversations
        min_conversations: Minimum conversations needed
        
    Returns:
        List[str]: List of formatted conversation strings, or empty list if insufficient data
    """
    from app.models.UserConversation import UserConversation
    from app.db.database import get_db
    
    print(f"🔍 Fetching conversations for user {user_id} (last {days} days)")
    
    # Properly manage database session
    db_generator = get_db()
    db = next(db_generator)
    
    try:
        # Calculate the date from N days ago
        days_ago = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get conversations for the specific user
        conversations = (
            db.query(UserConversation)
            .filter(
                UserConversation.user_id == user_id,
                UserConversation.conversation_start >= days_ago,
            )
            .order_by(UserConversation.conversation_start.desc())
            .all()
        )
        
        if not conversations:
            print(f"❌ No conversations found for user {user_id} in the last {days} days.")
            return []

        if len(conversations) < min_conversations:
            print(f"⚠️  Only {len(conversations)} conversations found. Need at least {min_conversations} for quality starters.")
            return []

        print(f"✅ Found {len(conversations)} conversations for user {user_id}")
        
        # Group conversations into dialogue sessions
        # For now, we'll create one large conversation string with all interactions
        conv_history_formatted = "\n".join([
            f"{'agent' if c.is_agent else 'user'}: {c.conversation}"
            for c in conversations
        ])
        
        # Return as a list (workflow expects List[str])
        return [conv_history_formatted]
        
    except Exception as e:
        print(f"❌ Error retrieving conversations: {str(e)}")
        return []
    finally:
        # Close database connection properly
        try:
            next(db_generator)  # This triggers the finally block in get_db()
        except StopIteration:
            pass  # Expected when generator completes


# Example usage
if __name__ == "__main__":
    view_user_conversations()
        