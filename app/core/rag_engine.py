import json
import time
from typing import List, Dict, Any, Optional
from app.config import settings

class MockRAGEngine:
    """Mock RAG engine for development without heavy ML dependencies"""
    
    def __init__(self):
        self.mock_documents = [
            {
                "id": "doc_1",
                "title": "Home Loan Eligibility Criteria",
                "content": "For salaried individuals, minimum monthly income should be ₹25,000. For self-employed, minimum annual income should be ₹3 lakh. Age should be between 21-65 years.",
                "source": "loan_guidelines.pdf",
                "confidence": 0.85
            },
            {
                "id": "doc_2", 
                "title": "Karnataka Stamp Duty",
                "content": "In Karnataka, stamp duty for home purchase is 5% for women and 6% for men. Additional 1% registration charges apply.",
                "source": "karnataka_stamp_duty.pdf",
                "confidence": 0.92
            },
            {
                "id": "doc_3",
                "title": "Required Documents",
                "content": "Required documents include: Salary slips (3 months), Bank statements (6 months), Form 16, Identity proof (Aadhaar/PAN), Property documents.",
                "source": "documentation_guide.pdf", 
                "confidence": 0.88
            }
        ]
    
    async def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Mock document retrieval with simple keyword matching"""
        
        start_time = time.time()
        
        # Simple keyword-based matching
        query_lower = query.lower()
        relevant_docs = []
        
        for doc in self.mock_documents:
            # Check if any query terms appear in document content or title
            content_lower = doc["content"].lower() + " " + doc["title"].lower()
            
            # Simple scoring based on keyword overlap
            score = 0
            query_words = query_lower.split()
            for word in query_words:
                if word in content_lower:
                    score += 1
            
            if score > 0:
                doc_copy = doc.copy()
                doc_copy["relevance_score"] = score / len(query_words)
                relevant_docs.append(doc_copy)
        
        # Sort by relevance
        relevant_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "documents": relevant_docs[:top_k],
            "processing_time_ms": processing_time,
            "total_docs": len(relevant_docs)
        }
    
    def add_document(self, doc_id: str, title: str, content: str, source: str):
        """Add a document to the mock index"""
        new_doc = {
            "id": doc_id,
            "title": title,
            "content": content,
            "source": source,
            "confidence": 0.80
        }
        self.mock_documents.append(new_doc)

# For now, use mock RAG engine
rag_engine = MockRAGEngine()