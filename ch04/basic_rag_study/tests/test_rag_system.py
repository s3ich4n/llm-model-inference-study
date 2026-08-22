#!/usr/bin/env python3
"""
Simple test cases for the RAG System

This module contains basic tests for the RAG system functionality
using real OpenAI API calls and actual PDF files from knowledge_files folder.
"""

import sys
import unittest
from pathlib import Path

from containers import Container


class TestRAGSystemReal(unittest.TestCase):
    """Real test cases for RAGSystem class using actual OpenAI API and PDF files."""
    
    def setUp(self):
        """Set up test fixtures."""
        # 컨테이너가 .env를 읽는다. 키가 없으면 load_config()가 ValueError를 던진다.
        self.container = Container()
        try:
            self.config = self.container.config()
        except ValueError as e:
            self.skipTest(str(e))
        
        # Check if knowledge_files folder exists
        self.knowledge_folder = Path(self.config.knowledge_folder)
        if not self.knowledge_folder.exists():
            self.skipTest("knowledge_files folder not found. Please ensure PDF files are in the knowledge_files folder.")
        
        # Initialize RAG system
        try:
            self.rag_system = self.container.rag_system()
            print("✅ RAG system initialized successfully")
        except Exception as e:
            self.skipTest(f"Failed to initialize RAG system: {e}")
    
    def test_config_loading(self):
        """Test that configuration loads correctly."""
        config = self.config
        
        # Test that required config values exist
        self.assertIsNotNone(config.llm_model)
        self.assertIsNotNone(config.embedding_model)
        self.assertIsNotNone(config.knowledge_folder)
        self.assertIsNotNone(config.chunk_size)
        self.assertIsNotNone(config.chunk_overlap)
        
        print(f"✅ Config loaded: LLM={config.llm_model}, Embedding={config.embedding_model}")
        print(f"✅ Knowledge folder: {config.knowledge_folder}")
        print(f"✅ Chunk size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
    
    def test_pdf_files_exist(self):
        """Test that PDF files exist in the knowledge_files folder."""
        pdf_files = list(self.knowledge_folder.glob("*.pdf"))
        pdf_files.extend(self.knowledge_folder.glob("*.PDF"))
        
        self.assertGreater(len(pdf_files), 0, "No PDF files found in knowledge_files folder")
        
        print(f"✅ Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"   📄 {pdf_file.name}")
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        # Test identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = self.rag_system.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Test orthogonal vectors
        vec3 = [0.0, 1.0, 0.0]
        similarity_orthogonal = self.rag_system.cosine_similarity(vec1, vec3)
        self.assertAlmostEqual(similarity_orthogonal, 0.0, places=5)
        
        # Test opposite vectors
        vec4 = [-1.0, 0.0, 0.0]
        similarity_opposite = self.rag_system.cosine_similarity(vec1, vec4)
        self.assertAlmostEqual(similarity_opposite, -1.0, places=5)
        
        print("✅ Cosine similarity calculations work correctly")
    
    def test_text_splitting(self):
        """Test text splitting functionality."""
        test_text = "This is a test document with multiple sentences. It should be split into chunks based on token count."
        
        # Test the actual text splitting method
        chunks = self.rag_system._split_text(test_text)
        
        # Verify chunks were created
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        print(f"✅ Text splitting created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: {chunk[:50]}...")
    
    def test_embedding_generation(self):
        """Test embedding generation with real OpenAI API."""
        test_texts = ["This is a test document about artificial intelligence."]
        
        try:
            embeddings = self.rag_system.get_embeddings(test_texts)
            
            # Verify embeddings were generated
            self.assertIsInstance(embeddings, list)
            self.assertEqual(len(embeddings), 1)
            self.assertIsInstance(embeddings[0], list)
            self.assertGreater(len(embeddings[0]), 0)
            
            print(f"✅ Generated embeddings with {len(embeddings[0])} dimensions")
            
        except Exception as e:
            self.fail(f"Failed to generate embeddings: {e}")
    
    def test_pdf_loading(self):
        """Test loading PDF files from knowledge_files folder."""
        try:
            documents = self.rag_system.load_pdfs()
            
            # Verify documents were loaded
            self.assertGreater(len(documents), 0)
            
            # Verify document structure
            for doc in documents:
                self.assertIn('content', doc)
                self.assertIn('source', doc)
                self.assertIn('file_path', doc)
                self.assertIn('chunk_id', doc)
                self.assertIsInstance(doc['content'], str)
                self.assertIsInstance(doc['source'], str)
                self.assertIsInstance(doc['chunk_id'], int)
            
            print(f"✅ Loaded {len(documents)} document chunks")
            for doc in documents[:3]:  # Show first 3 documents
                print(f"   📄 {doc['source']} (chunk {doc['chunk_id']}): {doc['content'][:50]}...")
            
        except Exception as e:
            self.fail(f"Failed to load PDF files: {e}")
    
    def test_vector_database_building(self):
        """Test building the vector database."""
        try:
            # Build vector database
            self.rag_system.build_vector_db()
            
            # Verify vector database was built
            self.assertGreater(len(self.rag_system.documents), 0)
            self.assertGreater(len(self.rag_system.embeddings), 0)
            self.assertGreater(len(self.rag_system.metadata), 0)
            
            # Verify consistency
            self.assertEqual(len(self.rag_system.documents), len(self.rag_system.embeddings))
            self.assertEqual(len(self.rag_system.documents), len(self.rag_system.metadata))
            
            print(f"✅ Vector database built with {len(self.rag_system.documents)} documents")
            print(f"✅ Generated {len(self.rag_system.embeddings)} embeddings")
            
        except Exception as e:
            self.fail(f"Failed to build vector database: {e}")
    
    def test_search_functionality(self):
        """Test search functionality with real data."""
        # First build the vector database
        self.rag_system.build_vector_db()
        
        # Test search queries
        test_queries = [
            "artificial intelligence",
            "machine learning", 
            "database queries",
            "5-level paging"
        ]
        
        for query in test_queries:
            try:
                results = self.rag_system.search(query, k=3)
                
                # Verify search results
                self.assertIsInstance(results, list)
                self.assertGreater(len(results), 0)
                
                # Verify result structure
                for result in results:
                    self.assertIn('content', result)
                    self.assertIn('metadata', result)
                    self.assertIn('score', result)
                    self.assertIn('source', result['metadata'])
                    self.assertIn('chunk_id', result['metadata'])
                    self.assertIsInstance(result['score'], (int, float))
                
                print(f"✅ Search for '{query}' returned {len(results)} results (top score: {results[0]['score']:.4f})")
                
            except Exception as e:
                self.fail(f"Search failed for query '{query}': {e}")
    
    def test_context_generation(self):
        """Test context generation for queries."""
        # First build the vector database
        self.rag_system.build_vector_db()
        
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning concepts",
            "How do database queries work?"
        ]
        
        for query in test_queries:
            try:
                context = self.rag_system.get_context_for_query(query, k=2)
                
                # Verify context
                self.assertIsInstance(context, str)
                self.assertGreater(len(context), 0)
                self.assertIn("Document", context)
                self.assertIn("Source:", context)
                
                print(f"✅ Generated context for '{query[:30]}...' ({len(context)} characters)")
                
            except Exception as e:
                self.fail(f"Context generation failed for query '{query}': {e}")
    
    def test_full_workflow(self):
        """Test the complete RAG workflow end-to-end."""
        try:
            # 1. Load PDFs
            documents = self.rag_system.load_pdfs()
            self.assertGreater(len(documents), 0)
            print(f"✅ Step 1: Loaded {len(documents)} document chunks")
            
            # 2. Build vector database
            self.rag_system.build_vector_db()
            self.assertGreater(len(self.rag_system.documents), 0)
            print(f"✅ Step 2: Built vector database with {len(self.rag_system.documents)} documents")
            
            # 3. Search
            results = self.rag_system.search("artificial intelligence", k=3)
            self.assertGreater(len(results), 0)
            print(f"✅ Step 3: Search returned {len(results)} results")
            
            # 4. Generate context
            context = self.rag_system.get_context_for_query("machine learning")
            self.assertGreater(len(context), 0)
            print(f"✅ Step 4: Generated context ({len(context)} characters)")
            
            print("✅ Full RAG workflow completed successfully!")
            
        except Exception as e:
            self.fail(f"Full workflow failed: {e}")
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test search without building vector database
        with self.assertRaises(ValueError) as context:
            self.rag_system.search("test query")
        
        self.assertIn("Vector database not built", str(context.exception))
        print("✅ Correctly handles search without vector database")
        
        # Test context generation with no results
        self.rag_system.documents = []
        self.rag_system.embeddings = []
        self.rag_system.metadata = []
        
        context = self.rag_system.get_context_for_query("test query")
        self.assertEqual(context, "No relevant information found.")
        print("✅ Correctly handles context generation with no data")


def run_real_rag_tests():
    """Run real RAG system tests."""
    print("🧪 Running Real RAG System Tests")
    print("=" * 50)
    print("📝 Note: These tests use real OpenAI API calls and require:")
    print("   1. OpenAI API key in .env file")
    print("   2. PDF files in knowledge_files folder")
    print("   3. Internet connection for API calls")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestRAGSystemReal)
    )
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 Test Results: {result.testsRun} tests run")
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n⚠️  Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_real_rag_tests()
    sys.exit(0 if success else 1) 