from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from openai import OpenAI
from pypdf import PdfReader

from config import Settings
from logs import get_logger

logger = get_logger(__name__)

class RAGSystem:
    def __init__(self, settings: Settings, client: OpenAI):
        self.settings = settings
        self.client = client
        self.encoding = tiktoken.encoding_for_model(self.settings.llm_model)
        
        # In-memory storage for embeddings and documents
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
    def load_pdfs(self, folder_path: str | None = None) -> list[dict[str, Any]]:
        """Load all PDF files from the knowledge folder and split into chunks."""
        if folder_path is None:
            folder_path = self.settings.knowledge_folder
            
        documents = []
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}")
                
                # Read PDF
                reader = PdfReader(str(pdf_file))
                text_content = ""
                
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
                
                # Split into chunks
                chunks = self._split_text(text_content)
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "content": chunk,
                        "source": pdf_file.name,
                        "file_path": str(pdf_file),
                        "chunk_id": i
                    })
                
                logger.info(f"Successfully processed {pdf_file.name} into {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e!s}")
                continue
        
        return documents
    
    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks based on token count."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.settings.chunk_size - self.settings.chunk_overlap):
            chunk_tokens = tokens[i:i + self.settings.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
        
        return chunks
    
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts using OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings: {e!s}")
            raise
    
    def build_vector_db(self, force_rebuild: bool = False):
        """Build the vector database from PDF documents."""
        if self.documents and not force_rebuild:
            logger.info("Vector database already exists. Use force_rebuild=True to rebuild.")
            return
        
        logger.info("Building vector database...")
        
        # Load documents
        documents = self.load_pdfs()
        if not documents:
            logger.warning("No documents found to process")
            return
        
        # Get embeddings
        texts = [doc["content"] for doc in documents]
        embeddings = self.get_embeddings(texts)
        
        # Store in memory
        self.documents = documents
        self.embeddings = embeddings
        self.metadata = [{"source": doc["source"], "chunk_id": doc["chunk_id"]} for doc in documents]
        
        logger.info(f"Vector database built with {len(documents)} documents")
    
    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search the vector database for relevant documents."""
        if not self.documents:
            raise ValueError("Vector database not built. Call build_vector_db() first.")
        
        logger.info(f"Searching for query: {query}")
        
        # Get query embedding
        query_embedding = self.get_embeddings([query])[0]
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity and get top k
        similarities.sort(reverse=True)
        top_k_indices = [idx for _, idx in similarities[:k]]
        
        # Format results
        results = []
        for i, idx in enumerate(top_k_indices):
            results.append({
                "content": self.documents[idx]["content"],
                "metadata": self.metadata[idx],
                "score": similarities[i][0]
            })
        
        logger.info(f"Found {len(results)} relevant documents")
        return results
    
    def get_context_for_query(self, query: str, k: int = 5) -> str:
        """Get formatted context from search results."""
        # documents가 비어 있으면 search()가 ValueError를 던진다. 비어 있지
        # 않으면 상위 k개는 반드시 나오므로 결과가 빈 경우는 없다.
        results = self.search(query, k)

        context_parts = []
        for i, result in enumerate(results, 1):
            content = result["content"]
            source = result["metadata"]["source"]
            context_parts.append(f"Document {i} (Source: {source}):\n{content}\n")
        
        return "\n".join(context_parts)
    
    # def save_vector_db(self, filepath: str = "vector_db.json"):
    #     """Save vector database to file."""
    #     if not self.documents:
    #         logger.warning("No vector database to save")
    #         return
        
    #     data = {
    #         "documents": self.documents,
    #         "embeddings": self.embeddings,
    #         "metadata": self.metadata
    #     }
        
    #     with open(filepath, 'w') as f:
    #         json.dump(data, f, indent=2)
        
    #     logger.info(f"Vector database saved to {filepath}")
    
    # def load_vector_db(self, filepath: str = "vector_db.json"):
    #     """Load vector database from file."""
    #     if not os.path.exists(filepath):
    #         logger.warning(f"Vector database file {filepath} not found")
    #         return
        
    #     with open(filepath, 'r') as f:
    #         data = json.load(f)
        
    #     self.documents = data["documents"]
    #     self.embeddings = data["embeddings"]
    #     self.metadata = data["metadata"]
        
    #     logger.info(f"Vector database loaded from {filepath} with {len(self.documents)} documents") 