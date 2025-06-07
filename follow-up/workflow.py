from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import logging

import importlib.util
import sys
import os

# Import chat segmenter rater
spec1 = importlib.util.spec_from_file_location(
    "chat_segmenter_rater", 
    os.path.join(os.path.dirname(__file__), "agents", "chat-segmenter-rater.py")
)
chat_segmenter_rater = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(chat_segmenter_rater)

# Import conversation starter generator
spec2 = importlib.util.spec_from_file_location(
    "conversation_starter_generator", 
    os.path.join(os.path.dirname(__file__), "agents", "conversation-starter-generator.py")
)
conversation_starter_generator = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(conversation_starter_generator)

# Extract the imports we need
make_agent_chat_segmenter_rater = chat_segmenter_rater.make_agent_chat_segmenter_rater
SegmenterRaterDeps = chat_segmenter_rater.SegmenterRaterDeps
ConversationSegment = chat_segmenter_rater.ConversationSegment

make_agent_conversation_starter_generator = conversation_starter_generator.make_agent_conversation_starter_generator
StarterGeneratorDeps = conversation_starter_generator.StarterGeneratorDeps
ConversationStarter = conversation_starter_generator.ConversationStarter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add console handler to see logging output
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(handler)


class ConversationFollowUpState(TypedDict):
    user_id: str
    raw_conversations: List[str]
    conversation_segments: List[ConversationSegment]
    top_segments: List[ConversationSegment]
    conversation_starters: List[ConversationStarter]


def get_conversation_followup_workflow(model_name="o3"):
    """Creates the conversation follow-up workflow with 2 agents"""
    
    agent_segmenter_rater = make_agent_chat_segmenter_rater(model_name=model_name)
    agent_starter_generator = make_agent_conversation_starter_generator(model_name=model_name)

    async def node_segment_and_rate(state: ConversationFollowUpState):
        """Node 1: Segment conversations and rate engagement"""
        logger.info("Running conversation segmentation and rating")

        # Process each conversation individually
        all_segments = []
        for i, conversation in enumerate(state["raw_conversations"]):
            logger.info(f"Processing conversation {i+1}/{len(state['raw_conversations'])}")
            
            deps = SegmenterRaterDeps(
                conversation=conversation
            )

            result = await agent_segmenter_rater.run("Please analyze this conversation.", deps=deps)
            segments = result.data.segments
            all_segments.extend(segments)

        logger.info(f"Created {len(all_segments)} conversation segments total")

        # Get top 3 segments by combined score (engagement + enjoyment) for better coverage
        top_segments = sorted(all_segments, key=lambda x: x.combined_score, reverse=True)[:3]
        
        return {
            "conversation_segments": all_segments,
            "top_segments": top_segments
        }

    async def node_generate_starters(state: ConversationFollowUpState):
        """Node 2: Generate conversation starters from top segments"""
        logger.info("Generating conversation starters")

        deps = StarterGeneratorDeps(
            top_segments=state["top_segments"]
        )

        result = await agent_starter_generator.run(deps=deps)
        starters = result.data

        logger.info(f"Generated {len(starters)} conversation starters")

        return {"conversation_starters": starters}

    async def node_finish(state: ConversationFollowUpState):
        """Final node: Display results"""
        print("\n" + "="*60)
        print("📊 FINAL RESULTS")
        print("="*60)
        print(f"👤 User ID: {state['user_id']}")
        print(f"📝 Processed {len(state['raw_conversations'])} conversations")
        print(f"🔍 Generated {len(state['conversation_segments'])} total segments")
        
        print("\n📈 TOP CONVERSATION SEGMENTS:")
        print("-" * 60)
        for i, segment in enumerate(state["top_segments"], 1):
            print(f"\n{i}. SEGMENT {segment.segment_id} (Score: {segment.combined_score}/20)")
            print(f"   📋 Topic: {segment.topic}")
            print(f"   🎯 Engagement: {segment.engagement_score}/10 - {segment.engagement_justification}")
            print(f"   😊 Enjoyment: {segment.enjoyment_score}/10 - {segment.enjoyment_justification}")
            print(f"   💬 Content: {segment.content[:150]}...")
            
            # Show more detail for top 5 segments
            if i <= 5:
                print(f"   🎭 Tone: {segment.tone}")
                print(f"   🧭 Direction: {segment.conversation_direction}")
        
        print("\n🎯 GENERATED CONVERSATION STARTERS:")
        print("-" * 60)
        # Show top 15 starters (instead of all 30 to keep output manageable)
        for starter in state["conversation_starters"][:15]:
            print(f"\n{starter.rank}. {starter.starter}")
            print(f"   💡 Context: {starter.context}")
            
        if len(state["conversation_starters"]) > 15:
            print(f"\n... and {len(state['conversation_starters']) - 15} more conversation starters (ranks 16-30)")

        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        
        return state

    # Build the workflow graph
    builder = StateGraph(ConversationFollowUpState)

    # Add nodes
    builder.add_node("node_segment_and_rate", node_segment_and_rate)
    builder.add_node("node_generate_starters", node_generate_starters)
    builder.add_node("node_finish", node_finish)

    # Add edges (sequential flow)
    builder.add_edge(START, "node_segment_and_rate")
    builder.add_edge("node_segment_and_rate", "node_generate_starters")
    builder.add_edge("node_generate_starters", "node_finish")
    builder.add_edge("node_finish", END)

    # Compile with memory
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph


# Example usage
if __name__ == "__main__":
    import asyncio
    # Import chat-retrieval using importlib since filename has hyphen
    import importlib.util
    spec = importlib.util.spec_from_file_location("chat_retrieval", "chat-retrieval.py")
    chat_retrieval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chat_retrieval)
    get_user_conversations_for_workflow = chat_retrieval.get_user_conversations_for_workflow
    
    async def test_workflow():
        print("🚀 Starting Intelligent Conversation Follow-Up System")
        print("=" * 60)
        
        # Get real user conversations from database
        user_id = "Uojc2mPrSHXzWaGkqRtBsT8xCYb2"  # Default user ID
        print(f"🔍 Fetching conversations for user: {user_id}")
        
        real_conversations = get_user_conversations_for_workflow(
            user_id=user_id,
            days=30,
            min_conversations=5
        )
        
        if not real_conversations:
            print("❌ No sufficient conversation data found. Using sample data for testing.")
            # Fallback to sample data if no real conversations
            real_conversations = [
                "user: What is a turnpike? I've been wondering about this for a while.\nagent: A turnpike is a toll road, especially in the northeastern United States. They're called turnpikes because historically, they had gates that would turn to let traffic through after paying the toll.\nuser: Oh that's really interesting! I never knew the etymology. What about blind spots when driving?\nagent: Blind spots are areas around your vehicle that you cannot see in your mirrors or through your windows directly. They're typically located to the sides and rear of your vehicle.\nuser: Got it, that's very helpful! I'm learning so much about driving today."
            ]
        
        print(f"📝 Processing {len(real_conversations)} conversation(s)")
        print("=" * 60)
        
        workflow = get_conversation_followup_workflow()
        
        config = {
            "configurable": {"thread_id": f"real-{user_id}"}
        }
        
        initial_state = ConversationFollowUpState(
            user_id=user_id,
            raw_conversations=real_conversations,
            conversation_segments=[],
            top_segments=[],
            conversation_starters=[]
        )
        
        print("🤖 Running AI workflow...")
        result = await workflow.ainvoke(initial_state, config=config)
        
        print("\n" + "=" * 60)
        print("✅ Workflow completed successfully!")
        print("=" * 60)
        
        return result
    
    # Run test with real data
    asyncio.run(test_workflow()) 