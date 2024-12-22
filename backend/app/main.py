from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import algorithms, networks, simulation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(algorithms.router)
app.include_router(networks.router)
app.include_router(simulation.router)
