import uvicorn
from fastapi import FastAPI
from app.core.config import settings
from app.routes import solve, graph

app = FastAPI()

app.include_router(solve.router, prefix="/solve")
app.include_router(graph.router, prefix="/graph")

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
