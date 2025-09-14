from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter(prefix="/eligibility", tags=["eligibility"])

class EligibilityRequest(BaseModel):
    monthly_income: float = Field(..., gt=0, description="Net monthly income")
    monthly_obligations: float = Field(0, ge=0, description="Existing EMIs/Bills")
    roi: float = Field(..., gt=0, description="Annual interest rate in % (e.g. 9.5)")
    tenure_months: int = Field(..., gt=0, description="Tenure in months")
    loan_amount: Optional[float] = Field(None, gt=0, description="Proposed loan amount (optional)")
    foir_cap: float = Field(0.45, gt=0, lt=1, description="FOIR cap as fraction (e.g. 0.45 for 45%)")

class EligibilityResponse(BaseModel):
    emi: Optional[float] = None
    foir: Optional[float] = None
    eligible: Optional[bool] = None
    foir_cap: float
    max_eligible_emi: float
    max_eligible_loan: float
    notes: list[str]

def _emi_from_principal(P: float, annual_rate_pct: float, n_months: int) -> float:
    r = (annual_rate_pct / 100.0) / 12.0
    if r == 0:
        return P / n_months
    factor = (1 + r) ** n_months
    return P * r * factor / (factor - 1)

def _principal_from_emi(E: float, annual_rate_pct: float, n_months: int) -> float:
    r = (annual_rate_pct / 100.0) / 12.0
    if r == 0:
        return E * n_months
    factor = (1 + r) ** n_months
    return E * (factor - 1) / (r * factor)

@router.post("/calculate", response_model=EligibilityResponse)
def calculate(req: EligibilityRequest):
    try:
        # Max EMI allowed by FOIR cap
        max_emi = max(0.0, req.monthly_income * req.foir_cap - req.monthly_obligations)
        max_loan = _principal_from_emi(max_emi, req.roi, req.tenure_months)

        notes = [
            f"FOIR cap = {int(req.foir_cap*100)}%. Max eligible EMI = income*cap - obligations.",
            "Formulas: EMI = P*r*(1+r)^n / ((1+r)^n - 1); P from EMI is the inverse.",
        ]

        if req.loan_amount:
            emi = _emi_from_principal(req.loan_amount, req.roi, req.tenure_months)
            foir = (req.monthly_obligations + emi) / req.monthly_income
            eligible = foir <= req.foir_cap
            return EligibilityResponse(
                emi=round(emi, 2),
                foir=round(foir, 4),
                eligible=eligible,
                foir_cap=req.foir_cap,
                max_eligible_emi=round(max_emi, 2),
                max_eligible_loan=round(max_loan, 2),
                notes=notes,
            )
        else:
            # No proposed loan; return max capacity
            return EligibilityResponse(
                emi=None,
                foir=None,
                eligible=None,
                foir_cap=req.foir_cap,
                max_eligible_emi=round(max_emi, 2),
                max_eligible_loan=round(max_loan, 2),
                notes=notes + ["No loan_amount provided; returning maximum eligibility only."],
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Eligibility calculation failed: {e}")
