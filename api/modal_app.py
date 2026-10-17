"""Serve the Six Eyes API on Modal at a stable URL.

Deploy from the repo root:
    .venv/Scripts/modal deploy api/modal_app.py
URL: https://<workspace>--six-eyes-api-web.modal.run
"""
from pathlib import Path

import modal

app = modal.App("six-eyes-api")
db_volume = modal.Volume.from_name("six-eyes-db", create_if_missing=True)


def fetch_face_model():
    from insightface.app import FaceAnalysis

    FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                 providers=["CPUExecutionProvider"])


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")  # OpenCV's runtime libraries
    .pip_install_from_requirements(str(Path(__file__).with_name("requirements.txt")))
    .run_function(fetch_face_model)  # bake the buffalo_l weights into the image
    .env({"DB_PATH": "/data/six_eyes.db"})
    .add_local_python_source("main", "x", "signals", "analyze")
    .add_local_file(Path(__file__).with_name("data") / "keywords.json", "/root/data/keywords.json")
)


# ponytail: one container is the only SQLite writer; Modal commits the Volume in the
# background every few seconds and on shutdown. Move to Postgres before scaling out.
@app.function(image=image, volumes={"/data": db_volume}, secrets=[modal.Secret.from_name("six-eyes-x")],
              cpu=2, memory=2048, max_containers=1, scaledown_window=600, timeout=120)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from main import app as api, face_model

    face_model()  # load the model when the container starts, not on the first enroll
    return api
