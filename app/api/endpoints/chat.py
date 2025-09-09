from fastapi import APIRouter, HTTPException
from app.models.request_models import ChatRequest
from app.models.response_models import ChatResponse, Citation
from app.core.llm_client import llm_client
from app.core.rag_engine import rag_engine
import uuid
import time

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for AI assistance"""
    
    try:
        start_time = time.time()
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Retrieve relevant documents
        retrieval_result = await rag_engine.retrieve_documents(
            request.message, 
            top_k=5
        )
        
        # Convert documents to citations
        citations = []
        context_docs = []
        
        for doc in retrieval_result["documents"]:
            citation = Citation(
                id=doc["id"],
                title=doc["title"], 
                content=doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                source=doc["source"],
                confidence=doc.get("confidence", 0.8)
            )
            citations.append(citation)
            context_docs.append(doc)
        
        # Generate LLM response
        llm_result = await llm_client.generate_response(
            request.message,
            context_docs=context_docs
        )
        
        # Determine next steps based on query
        next_steps = _generate_next_steps(request.message)
        
        # Check if escalation needed
        should_escalate = _should_escalate(request.message, llm_result.get("success", True))
        
        processing_time = (time.time() - start_time) * 1000
        
        return ChatResponse(
            response=llm_result["response"],
            citations=citations,
            next_steps=next_steps,
            confidence=0.85,
            should_escalate=should_escalate,
            session_id=session_id,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

def _generate_next_steps(query: str) -> list:
    """Generate contextual next steps"""
    query_lower = query.lower()
    
    if "eligible" in query_lower:
        return [
            "Schedule a consultation with our loan advisor",
            "Gather required documentation",
            "Get pre-approval certificate"
        ]
    elif "document" in query_lower:
        return [
            "Prepare your income documents",
            "Collect property papers", 
            "Visit nearest branch for verification"
        ]
    elif "rate" in query_lower:
        return [
            "Compare different loan schemes",
            "Check for special offers",
            "Calculate EMI using our calculator"
        ]
    else:
        return [
            "Explore our loan products",
            "Use our eligibility calculator",
            "Speak with a loan advisor"
        ]

def _should_escalate(query: str, llm_success: bool) -> bool:
    """Determine if human escalation is needed"""
    if not llm_success:
        return True
    
    escalation_keywords = ["complaint", "problem", "issue", "dissatisfied", "cancel"]
    return any(keyword in query.lower() for keyword in escalation_keywords)