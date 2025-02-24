import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import graphs, solvers

app = FastAPI(title="Optimal License Distribution API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graphs.router, prefix="/api/graphs", tags=["Graphs"])
app.include_router(solvers.router, prefix="/api/solvers", tags=["Solvers"])

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.uvicorn_host,
        port=settings.uvicorn_port,
        reload=True,
    )
