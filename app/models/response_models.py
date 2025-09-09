from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Citation(BaseModel):
    id: str
    title: str
    content: str
    source: str
    confidence: float

class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]
    next_steps: List[str]
    confidence: float
    should_escalate: bool
    session_id: str
    processing_time_ms: float

class EligibilityResponse(BaseModel):
    max_emi: float
    estimated_loan_amount: float
    foir_used: float
    monthly_surplus: float
    eligibility_status: str
    disclaimer: str

class FeedbackResponse(BaseModel):
    success: bool
    message: str

class HealthResponse(BaseModel):
    status: str
    version: str
    dependencies: Dict[str, str]
    uptime_seconds: float