from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.models.database import init_db

settings = get_settings()

app = FastAPI(
    title=settings.app_name, description="Hyperliquid Trader Analysis Platform API", version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    init_db()


@app.get("/")
async def root():
    return {"name": settings.app_name, "version": "0.1.0", "docs": "/docs"}
