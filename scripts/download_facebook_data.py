import tarfile
import urllib.request
from pathlib import Path

SNAP_URL = "https://snap.stanford.edu/data/facebook.tar.gz"
DATA_DIR = Path("data/facebook")

def download_facebook_data() -> None:
    """Download and extract Facebook ego-network dataset from SNAP."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / "facebook.tar.gz"

    if not archive_path.exists():
        print(f"Pobieranie {SNAP_URL}...")
        urllib.request.urlretrieve(SNAP_URL, archive_path)
    else:
        print(f"{archive_path} już istnieje, pomijam pobieranie.")

    print("Rozpakowywanie archiwum...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print(f"Dane zapisane w: {DATA_DIR.resolve()}")

if __name__ == "__main__":
    download_facebook_data()
