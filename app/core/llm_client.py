import asyncio
import time
from typing import List, Dict, Any, Optional
from app.config import settings
from app.core.security import redactor
import json

class MockLLMClient:
    """Mock LLM client for development when Gemini is not available"""
    
    def __init__(self):
        self.mock_responses = {
            "eligibility": "Based on your income and FOIR ratio, you appear eligible for a home loan. Please consult with our loan officer for detailed assessment.",
            "documentation": "For home loan in Karnataka, you'll need income proof, identity documents, property papers, and bank statements. Specific requirements vary by loan type.",
            "rates": "Current home loan interest rates range from 8.5% to 11.5% depending on your profile and loan amount. Visit our branch for personalized rates.",
            "default": "Thank you for your query. I'll help you with your home loan questions. Could you please provide more specific details about what you'd like to know?"
        }
    
    async def generate_response(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]] = None,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """Generate mock response"""
        
        start_time = time.time()
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Simple keyword matching for mock responses
        query_lower = query.lower()
        if "eligible" in query_lower or "income" in query_lower:
            response = self.mock_responses["eligibility"]
        elif "document" in query_lower or "papers" in query_lower:
            response = self.mock_responses["documentation"] 
        elif "rate" in query_lower or "interest" in query_lower:
            response = self.mock_responses["rates"]
        else:
            response = self.mock_responses["default"]
        
        # Add context information if available
        if context_docs:
            response += f"\n\nBased on {len(context_docs)} relevant document(s) from our knowledge base."
        
        # Redact PII from response
        clean_response = redactor.redact_text(response)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "response": clean_response,
            "processing_time_ms": processing_time,
            "success": True
        }

class GeminiClient:
    """Real Gemini client - requires API key"""
    
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            self.available = True
        except ImportError:
            self.available = False
        
    async def generate_response(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]] = None,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """Generate response with Gemini"""
        
        if not self.available:
            return {"response": "Gemini LLM not available", "processing_time_ms": 0, "success": False}
            
        start_time = time.time()
        
        # Build prompt with context
        prompt = self._build_prompt(query, context_docs, system_prompt)
        
        try:
            response = await asyncio.wait_for(
                self._generate_async(prompt),
                timeout=settings.LLM_TIMEOUT
            )
            
            # Redact PII from response
            clean_response = redactor.redact_text(response)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "response": clean_response,
                "processing_time_ms": processing_time,
                "success": True
            }
            
        except asyncio.TimeoutError:
            return {
                "response": "I'm experiencing high load. Please try again in a moment.",
                "processing_time_ms": settings.LLM_TIMEOUT * 1000,
                "success": False
            }
    
    def _build_prompt(self, query: str, context_docs: List[Dict], system_prompt: str = None) -> str:
        """Build prompt with context and citations"""
        
        base_prompt = system_prompt or """You are an AI assistant for a home loan company in India. 
Provide helpful, accurate information about home loans, documentation, and eligibility.

IMPORTANT RULES:
1. Always cite sources using [1], [2] format when using provided documents
2. Never hallucinate policy details or specific rates
3. If uncertain, say so and suggest speaking with an advisor
4. Focus on being helpful while staying within your knowledge
5. Provide 1-3 concrete next steps at the end
6. Keep responses concise but complete"""

        if context_docs:
            context_text = "\n\nRELEVANT DOCUMENTS:\n"
            for i, doc in enumerate(context_docs, 1):
                context_text += f"[{i}] {doc.get('title', 'Document')}: {doc.get('content', '')}\n"
            prompt = f"{base_prompt}\n{context_text}\n\nQUERY: {query}\n\nRESPONSE:"
        else:
            prompt = f"{base_prompt}\n\nQUERY: {query}\n\nRESPONSE:"
            
        return prompt
    
    async def _generate_async(self, prompt: str) -> str:
        """Async wrapper for generation"""
        response = self.model.generate_content(prompt)
        return response.text

# Use mock client by default, can switch to real Gemini when API key is provided
try:
    llm_client = GeminiClient() if settings.GEMINI_API_KEY else MockLLMClient()
except:
    llm_client = MockLLMClient()