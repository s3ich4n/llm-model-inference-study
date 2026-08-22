#!/usr/bin/env python3
"""
Example usage of the PDF Knowledge Agent

This script demonstrates how to use the agent for querying
information from PDF files in the knowledge folder.
"""


from containers import container


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")
    
    # Initialize the agent
    agent = container.agent()
    
    # Build knowledge base
    print("Building knowledge base...")
    agent.build_knowledge_base()
    
    # Example queries
    queries = [
        "What is 5-level paging?",
        "Explain database queries and data mining",
        "What are Patricia tries?",
        "Tell me about Standard Annotation Language (SAL)",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        result = agent.process_query(query)
        
        if result["success"]:
            print(f"✅ Response: {result['final_response'][:200]}...")
            if result["plan"]:
                print(f"📋 Plan: {result['plan']['reasoning']}")
        else:
            print(f"❌ Error: {result['final_response']}")

def example_custom_user_profile():
    """Example with custom user profile."""
    print("\n=== Custom User Profile Example ===")
    
    # Create agent with custom user profile
    custom_profile = {
        "expertise_level": "beginner",
        "background": "business",
        "preferred_detail_level": "high",
    }
    
    agent = container.agent(user_profile=custom_profile)
    
    # Build knowledge base
    agent.build_knowledge_base()
    
    query = "What is 5-level paging and how does it work?"
    print(f"🔍 Query: {query}")
    print(f"👤 User Profile: {custom_profile}")
    
    result = agent.process_query(query)
    
    if result["success"]:
        print(f"✅ Response: {result['final_response']}")
    else:
        print(f"❌ Error: {result['final_response']}")

def example_direct_search():
    """Example of direct knowledge base search."""
    print("\n=== Direct Search Example ===")
    
    agent = container.agent()
    agent.build_knowledge_base()
    
    query = "database queries"
    print(f"🔍 Searching for: {query}")
    
    # Direct search in knowledge base
    search_results = agent.search_knowledge_base(query, k=3)
    
    print(f"Found {len(search_results)} relevant documents:")
    for i, result in enumerate(search_results, 1):
        print(f"\n📄 Document {i}:")
        print(f"   Source: {result['metadata']['source']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Content: {result['content'][:150]}...")

def example_system_status():
    """Example of checking system status."""
    print("\n=== System Status Example ===")
    
    agent = container.agent()
    agent.build_knowledge_base()
    status = agent.get_system_status()
    
    print("📊 System Status:")
    for component, info in status.items():
        print(f"  {component}: {info}")

def example_different_actions():
    """Example of different action types."""
    print("\n=== Different Actions Example ===")
    
    agent = container.agent()
    agent.build_knowledge_base()
    
    # Test different types of queries
    queries = [
        ("What is 5-level paging?", "Information query"),
        ("Summarize the key points about database queries", "Summary query"),
        ("Analyze the benefits and drawbacks of Patricia tries", "Analysis query"),
    ]
    
    for query, query_type in queries:
        print(f"\n🔍 {query_type}: {query}")
        result = agent.process_query(query)
        
        if result["success"]:
            print(f"✅ Response: {result['final_response'][:150]}...")
            if result["plan"]:
                print(f"📋 Actions: {result['plan']['plan']}")
        else:
            print(f"❌ Error: {result['final_response']}")

def main():
    """Run all examples."""
    container.init_resources()

    print("🤖 PDF Knowledge Agent - Example Usage")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_user_profile()
        example_direct_search()
        example_system_status()
        example_different_actions()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e!s}")
        print("\n💡 Make sure you have:")
        print("   1. Installed all requirements: uv sync")
        print("   2. Set up your OpenAI API key in .env file")
        print("   3. PDF files in the knowledge/ folder")

if __name__ == "__main__":
    main() 