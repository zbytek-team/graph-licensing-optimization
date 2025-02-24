from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    individual_license_cost: float = 10.0
    group_license_cost: float = 18.0
    group_license_size: int = 6
    uvicorn_host: str = "0.0.0.0"
    uvicorn_port: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()
