from typing import Any

from actions import ActionExecutor
from config import Settings
from llm_manager import LLMManager
from logs import get_logger
from planner import Planner
from rag_system import RAGSystem

logger = get_logger(__name__)

class Agent:
    """구성 요소를 직접 만들지 않고 컨테이너에서 받아 쓰는 에이전트."""

    def __init__(
        self,
        settings: Settings,
        rag_system: RAGSystem,
        llm_manager: LLMManager,
        planner: Planner,
        action_executor: ActionExecutor,
        user_profile: dict[str, Any] | None = None,
    ):
        self.settings = settings
        self.rag_system = rag_system
        self.llm_manager = llm_manager
        self.planner = planner
        self.action_executor = action_executor
        self.user_profile = user_profile or dict(self.settings.default_user_profile)

        logger.info("Agent initialized successfully")
    
    def build_knowledge_base(self, force_rebuild: bool = False):
        """Build the knowledge base from PDF files."""
        logger.info("Building knowledge base...")
        self.rag_system.build_vector_db(force_rebuild=force_rebuild)
        logger.info("Knowledge base built successfully")
    
    def process_query(self, query: str, use_planning: bool = True) -> dict[str, Any]:
        """Process a user query and return a comprehensive response."""
        logger.info(f"Processing query: {query}")
        
        try:
            if use_planning:
                # Create execution plan
                plan = self.planner.create_plan(query)
                action_sequence = self.planner.get_action_sequence(plan)
                
                logger.info(f"Execution plan: {plan}")
                logger.info(f"Action sequence: {action_sequence}")
                
                # Execute actions
                results = self._execute_action_sequence(query, action_sequence)
                
                return {
                    "query": query,
                    "plan": plan,
                    "results": results,
                    "final_response": results[-1] if results else "No response generated",
                    "success": True
                }
            else:
                # Direct execution without planning
                response = self.action_executor.query_rag_with_context(query)
                
                return {
                    "query": query,
                    "plan": None,
                    "results": [response],
                    "final_response": response,
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {e!s}")
            return {
                "query": query,
                "plan": None,
                "results": [],
                "final_response": f"Error processing query: {e!s}",
                "success": False,
                "error": str(e)
            }
    
    def _execute_action_sequence(self, query: str, action_sequence: list[str]) -> list[str]:
        """Execute a sequence of actions and return results."""
        results = []
        context = ""
        
        for i, action in enumerate(action_sequence):
            logger.info(f"Executing action {i+1}/{len(action_sequence)}: {action}")
            
            try:
                # Validate action prerequisites
                if not self.action_executor.validate_action_prerequisites(action):
                    logger.warning(f"Prerequisites not met for action: {action}")
                    continue
                
                # Execute action
                result = self.action_executor.execute_action(
                    action, query, context, self.user_profile
                )
                
                results.append(result)
                
                # Use the result as context for next action if available
                if result and len(result) > 50:  # Only use substantial results as context
                    context = result
                
                logger.info(f"Action {action} completed successfully")
                
            except Exception as e:
                logger.error(f"Error executing action {action}: {e!s}")
                results.append(f"Error in action {action}: {e!s}")
        
        return results
    
    def update_user_profile(self, new_profile: dict[str, Any]):
        """Update the user profile."""
        self.user_profile.update(new_profile)
        logger.info(f"Updated user profile: {self.user_profile}")
    
    def get_user_profile(self) -> dict[str, Any]:
        """Get the current user profile."""
        return self.user_profile.copy()
    
    def search_knowledge_base(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search the knowledge base directly."""
        return self.rag_system.search(query, k)
    
    def get_system_status(self) -> dict[str, Any]:
        """Get the current status of all system components."""
        return {
            "rag_system": {
                "documents_loaded": len(self.rag_system.documents),
                "embeddings_available": len(self.rag_system.embeddings),
                "knowledge_folder": self.settings.knowledge_folder
            },
            "llm_manager": {
                "model": self.settings.llm_model,
                "embedding_model": self.settings.embedding_model
            },
            "user_profile": self.user_profile,
            "available_actions": self.planner.available_actions
        }
    
    def interactive_mode(self):
        """Simple interactive mode - just process user queries."""
        print("🤖 PDF Knowledge Agent - Simple Interactive Mode")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                query = input("\n💬 Your question: ").strip()
                
                if query.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif not query:
                    continue
                
                # Process the query
                print("\n🔄 Processing...")
                result = self.process_query(query)
                
                if result["success"]:
                    print(f"\n✅ Response:\n{result['final_response']}")
                else:
                    print(f"\n❌ Error: {result['final_response']}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e!s}")
