#!/usr/bin/env python3
"""
Batch process all 10 users through the follow-up workflow.
Creates individual output files with all conversation starters.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
import logging

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import using importlib to handle hyphenated filenames
import importlib.util
import os

# Import workflow module
spec = importlib.util.spec_from_file_location("workflow", "workflow.py")
workflow_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow_module)

# Import chat-retrieval module
spec2 = importlib.util.spec_from_file_location("chat_retrieval", "chat-retrieval.py")
chat_retrieval_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(chat_retrieval_module)

# Get the functions we need
get_conversation_followup_workflow = workflow_module.get_conversation_followup_workflow
get_user_conversations_for_workflow = chat_retrieval_module.get_user_conversations_for_workflow

# User mapping from atarino-task-manager
USER_MAPPING = {
    "ada": "i7AqJRZqyeOsqXHtTP5fikVkJJv1",
    "el": "Uyta01R0ETd5XQT3mz6ikyGm0yf1", 
    "leonid": "Uojc2mPrSHXzWaGkqRtBsT8xCYb2",
    "kasophone": "T9jHW9fKfMab9thGY5Mj2wfzG6H2",
    "nurik": "dbWfZBQUOSdrZwXZnLk9ATnyrDp2",
    "qasim": "CAfWabjV1jU6VElGk4D0yRDNbU53",
    "amanzhol": "bkCpnPtejnZsspRb27P6apxJlP52",
    "jack": "APM8xRdIiZQIUmfLbcHePE0GFII2",
    "babushi": "0ZoBTbvPtrfQTUFIW9Yld5DpjWs1",
    "almas": "uFdp6q7Co4VXBboBZGOS9Pi2OuG3"
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def process_user(username: str, user_id: str) -> dict:
    """Process a single user through the workflow."""
    try:
        logger.info(f"Processing user: {username} ({user_id})")
        
        # Get user conversations
        conversations = get_user_conversations_for_workflow(user_id)
        if not conversations:
            logger.warning(f"No conversations found for {username}")
            return {"error": "No conversations found"}
        
        logger.info(f"Found {len(conversations)} conversations for {username}")
        
        # Create workflow instance
        workflow = get_conversation_followup_workflow()
        
        # Run workflow with correct state structure
        config = {"configurable": {"thread_id": f"batch-{username}-{user_id}"}}
        result = await workflow.ainvoke({
            "user_id": user_id,
            "raw_conversations": conversations,
            "conversation_segments": [],
            "top_segments": [],
            "conversation_starters": []
        }, config=config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {username}: {str(e)}")
        return {"error": str(e)}

def save_user_output(username: str, user_id: str, result: dict):
    """Save user results to a file."""
    filename = f"follow-ups/{username}-{user_id}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"=== CONVERSATION FOLLOW-UP ANALYSIS FOR {username.upper()} ===\n")
            f.write(f"User ID: {user_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            if "error" in result:
                f.write(f"ERROR: {result['error']}\n")
                return
            
            # Write conversation segments
            if "conversation_segments" in result:
                f.write("CONVERSATION SEGMENTS ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                segments = result["conversation_segments"]
                for i, segment in enumerate(segments, 1):
                    # Handle both dict and object access
                    if hasattr(segment, 'topic'):
                        f.write(f"\nSegment {i}:\n")
                        f.write(f"ID: {getattr(segment, 'segment_id', 'N/A')}\n")
                        f.write(f"Topic: {getattr(segment, 'topic', 'N/A')}\n")
                        f.write(f"Engagement Score: {getattr(segment, 'engagement_score', 'N/A')}\n")
                        f.write(f"Enjoyment Score: {getattr(segment, 'enjoyment_score', 'N/A')}\n")
                        f.write(f"Combined Score: {getattr(segment, 'combined_score', 'N/A')}\n")
                        f.write(f"Tone: {getattr(segment, 'tone', 'N/A')}\n")
                        f.write(f"Direction: {getattr(segment, 'conversation_direction', 'N/A')}\n")
                        f.write(f"Content: {getattr(segment, 'content', 'N/A')[:200]}...\n")
                    else:
                        f.write(f"\nSegment {i}: {segment}\n")
                f.write("\n" + "=" * 60 + "\n\n")
            
            # Write top segments
            if "top_segments" in result:
                f.write("TOP SEGMENTS FOR FOLLOW-UP:\n")
                f.write("-" * 40 + "\n")
                for i, segment in enumerate(result["top_segments"], 1):
                    if hasattr(segment, 'topic'):
                        f.write(f"\n#{i} - {getattr(segment, 'topic', 'N/A')}\n")
                        f.write(f"Combined Score: {getattr(segment, 'combined_score', 'N/A')}\n")
                        f.write(f"Engagement: {getattr(segment, 'engagement_score', 'N/A')} - {getattr(segment, 'engagement_justification', 'N/A')}\n")
                        f.write(f"Enjoyment: {getattr(segment, 'enjoyment_score', 'N/A')} - {getattr(segment, 'enjoyment_justification', 'N/A')}\n")
                        f.write(f"Content: {getattr(segment, 'content', 'N/A')[:200]}...\n")
                    else:
                        f.write(f"\n#{i}: {segment}\n")
                f.write("\n" + "=" * 60 + "\n\n")
            
            # Write conversation starters
            if "conversation_starters" in result:
                f.write("30 PERSONALIZED CONVERSATION STARTERS:\n")
                f.write("-" * 40 + "\n")
                starters = result["conversation_starters"]
                
                for i, starter in enumerate(starters, 1):
                    if hasattr(starter, 'starter'):
                        f.write(f"\n{i}. {getattr(starter, 'starter', 'N/A')}\n")
                        f.write(f"   Context: {getattr(starter, 'context', 'N/A')}\n")
                        f.write(f"   Rank: {getattr(starter, 'rank', 'N/A')}\n")
                    else:
                        f.write(f"\n{i}. {starter}\n")
                
                f.write(f"\n\nTOTAL: {len(starters)} conversation starters generated\n")
            
        logger.info(f"Saved output for {username} to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving output for {username}: {str(e)}")

async def main():
    """Main batch processing function."""
    logger.info("Starting batch processing of all users...")
    start_time = datetime.now()
    
    successful = 0
    failed = 0
    
    for username, user_id in USER_MAPPING.items():
        try:
            result = await process_user(username, user_id)
            save_user_output(username, user_id, result)
            
            if "error" not in result:
                successful += 1
                logger.info(f"✅ Successfully processed {username}")
            else:
                failed += 1
                logger.error(f"❌ Failed to process {username}: {result['error']}")
                
        except Exception as e:
            failed += 1
            logger.error(f"❌ Exception processing {username}: {str(e)}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"\n=== BATCH PROCESSING COMPLETE ===")
    logger.info(f"Total users: {len(USER_MAPPING)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Duration: {duration}")
    logger.info(f"Output files saved in: follow-ups/")

if __name__ == "__main__":
    asyncio.run(main()) 