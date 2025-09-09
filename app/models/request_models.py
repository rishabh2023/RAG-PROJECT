from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    session_id: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    user_type: Optional[str] = Field(None, description="salaried, self_employed, etc.")

class EligibilityRequest(BaseModel):
    monthly_income: float = Field(..., gt=0, description="Monthly income in INR")
    existing_obligations: float = Field(0, ge=0, description="Existing EMIs in INR")
    interest_rate: float = Field(..., gt=0, le=30, description="Interest rate percentage")
    tenure_years: int = Field(..., gt=0, le=30, description="Loan tenure in years")
    foir_cap: Optional[float] = Field(40.0, gt=0, le=100, description="FOIR cap percentage")

class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = Field(None, description="helpful, accurate, fast, etc.")