from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from app.config import settings
from app.api.endpoints import chat, eligibility
from app.core.security import verify_bearer_token
import time

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token"""
    if not verify_bearer_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    print(f"🚀 Starting {settings.PROJECT_NAME}")
    print(f"🔧 API Version: {settings.API_VERSION}")
    print(f"🌐 Environment: {'Development' if settings.USE_LOCAL_FAISS else 'Production'}")
    yield
    print("🛑 Shutting down application")

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI Customer Support Backend for Home Loans",
    version="1.0.0",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response



app.include_router(
    chat.router,
    prefix=settings.API_PREFIX,
    tags=["chat"],
    dependencies=[Depends(verify_token)]
)

app.include_router(
    eligibility.router,
    prefix=settings.API_PREFIX,
    tags=["eligibility"],
    dependencies=[Depends(verify_token)]
)



@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Loan Support Backend",
        "version": "1.0.0",
        "docs_url": f"{settings.API_PREFIX}/docs",
        "health_check": f"{settings.API_PREFIX}/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )