from sentence_transformers import SentenceTransformer
import numpy as np
import re
from rank_bm25 import BM25Okapi

class RetrievalSystem:
    def __init__(self):
        # Engine 1: The Vector Model (Meaning)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        
        # Engine 2: The BM25 Model (Keywords)
        self.bm25 = None
        self.tokenized_corpus = []
        
        # The Data
        self.chunks = []

    def _tokenize(self, text):
        """Helper function: Strips punctuation and makes everything lowercase for BM25."""
        return re.findall(r'\w+', text.lower())

    def embed_documents(self, chunks_with_metadata):
        self.chunks = chunks_with_metadata
        text_only = [c["text"] for c in chunks_with_metadata]
        
        print("1/2: Embedding documents for Vector Search...")
        self.embeddings = self.model.encode(text_only)
        
        print("2/2: Building BM25 Index for Keyword Search...")
        self.tokenized_corpus = [self._tokenize(doc) for doc in text_only]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print("Hybrid Database Ready!")

    def _normalize(self, scores):
        """Min-Max Normalization: Squashes any range of numbers to fit strictly between 0.0 and 1.0"""
        if len(scores) == 0: return scores
        s_min = np.min(scores)
        s_max = np.max(scores)
        
        # Prevent division by zero if all scores are identical
        if s_max == s_min: return np.zeros_like(scores) 
        
        return (scores - s_min) / (s_max - s_min)

    def retrieve(self, query, top_k=5, alpha=0.5):
        """
        alpha = 1.0 -> Pure Vector Search
        alpha = 0.0 -> Pure Keyword Search
        alpha = 0.5 -> 50/50 Hybrid Search
        """
        if self.embeddings is None or self.bm25 is None:
            return []
        
        # --- 1. GET VECTOR SCORES ---
        query_vector = self.model.encode([query])[0]
        vector_scores = np.dot(self.embeddings, query_vector)
        
        # --- 2. GET BM25 SCORES ---
        tokenized_query = self._tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # --- 3. NORMALIZE SCORES ---
        norm_vector = self._normalize(vector_scores)
        norm_bm25 = self._normalize(bm25_scores)
        
        # --- 4. THE HYBRID FUSION ---
        # We multiply the normalized scores by the alpha weights
        hybrid_scores = (alpha * norm_vector) + ((1 - alpha) * norm_bm25)
        
        # --- 5. SORT AND RETURN ---
        # Get the indices of the highest hybrid scores
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append(self.chunks[idx])
            
        return results