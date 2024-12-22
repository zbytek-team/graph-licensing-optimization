from fastapi import APIRouter

router = APIRouter(prefix="/algorithms", tags=["Algorithms"])


@router.get("/")
def get_available_algorithms():
    return ["Greedy", "Greedy Multi-Group", "ILP", "GA"]
