from fastapi import APIRouter, HTTPException
from app.models.request_models import EligibilityRequest
from app.models.response_models import EligibilityResponse

router = APIRouter()

@router.post("/eligibility", response_model=EligibilityResponse)
async def calculate_eligibility(request: EligibilityRequest):
    """Calculate loan eligibility based on income and FOIR"""
    
    try:
        # Calculate maximum EMI based on FOIR
        max_emi = (request.monthly_income * request.foir_cap / 100) - request.existing_obligations
        
        if max_emi <= 0:
            return EligibilityResponse(
                max_emi=0,
                estimated_loan_amount=0,
                foir_used=100.0,
                monthly_surplus=0,
                eligibility_status="Not Eligible",
                disclaimer="Your existing obligations exceed the FOIR limit. Please reduce existing EMIs or increase income."
            )
        
        # Calculate loan amount using EMI formula
        # EMI = P * r * (1+r)^n / ((1+r)^n - 1)
        # Rearranging: P = EMI * ((1+r)^n - 1) / (r * (1+r)^n)
        
        monthly_rate = request.interest_rate / (12 * 100)
        tenure_months = request.tenure_years * 12
        
        if monthly_rate == 0:
            estimated_loan_amount = max_emi * tenure_months
        else:
            power_factor = (1 + monthly_rate) ** tenure_months
            estimated_loan_amount = max_emi * (power_factor - 1) / (monthly_rate * power_factor)
        
        # Calculate actual FOIR used
        total_emi = max_emi + request.existing_obligations
        foir_used = (total_emi / request.monthly_income) * 100
        
        # Calculate monthly surplus
        monthly_surplus = request.monthly_income - total_emi
        
        # Determine eligibility status
        if estimated_loan_amount >= 500000:  # Minimum 5 lakh
            eligibility_status = "Eligible"
        elif estimated_loan_amount >= 100000:
            eligibility_status = "Conditionally Eligible"
        else:
            eligibility_status = "Not Eligible"
        
        return EligibilityResponse(
            max_emi=round(max_emi, 2),
            estimated_loan_amount=round(estimated_loan_amount, 2),
            foir_used=round(foir_used, 2),
            monthly_surplus=round(monthly_surplus, 2),
            eligibility_status=eligibility_status,
            disclaimer="This is an indicative calculation. Final eligibility depends on credit score, property value, and bank policies. Please consult our loan advisor for accurate assessment."
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Eligibility calculation failed: {str(e)}")