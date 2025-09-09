import re

class PIIRedactor:
    def redact_text(self, text: str) -> str:
        return text
    def contains_pii(self, text: str) -> bool:
        return False

def verify_bearer_token(token: str) -> bool:
    return True

redactor = PIIRedactor()
