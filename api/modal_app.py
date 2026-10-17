"""Serve the Six Eyes API on Modal at a stable URL.

Deploy from the repo root, on a machine that has the trained heads in data/models (ml/heads.py):
    .venv/Scripts/modal deploy api/modal_app.py
URL: https://<workspace>--six-eyes-api-web.modal.run
Needs two Modal secrets: six-eyes-x (X_BEARER_TOKEN) and huggingface (HF_TOKEN, for the gated DINOv3).
"""
from pathlib import Path

import modal

app = modal.App("six-eyes-api")
db_volume = modal.Volume.from_name("six-eyes-db", create_if_missing=True)
MODELS = Path(__file__).resolve().parents[1] / "data" / "models"  # git-ignored; made by ml/heads.py


def fetch_models():
    import lpips
    from diffusers import AutoencoderKL
    from insightface.app import FaceAnalysis
    from transformers import AutoImageProcessor, AutoModel, pipeline

    FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                 providers=["CPUExecutionProvider"])
    dino = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # same model ids as analyze.py and recon.py
    AutoModel.from_pretrained(dino)
    AutoImageProcessor.from_pretrained(dino)
    pipeline("image-classification", model="Falconsai/nsfw_image_detection")
    AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")
    lpips.LPIPS(net="vgg", verbose=False)  # downloads torchvision's VGG16 into the image too


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")  # OpenCV's runtime libraries
    .pip_install_from_requirements(str(Path(__file__).with_name("requirements.txt")))
    # Bake the face, DINOv3, NSFW, VAE and LPIPS weights into the image; the token is only needed for this step.
    .run_function(fetch_models, secrets=[modal.Secret.from_name("huggingface")])
    .env({"DB_PATH": "/data/six_eyes.db", "MODELS_DIR": "/root/models", "HF_HUB_OFFLINE": "1"})
    .add_local_python_source("main", "x", "signals", "analyze", "face", "crops", "recon")
    .add_local_file(Path(__file__).with_name("data") / "keywords.json", "/root/data/keywords.json")
    .add_local_file(MODELS / "dinov3_binary.joblib", "/root/models/dinov3_binary.joblib")
    .add_local_file(MODELS / "dinov3_multi.joblib", "/root/models/dinov3_multi.joblib")
)


# ponytail: one container is the only SQLite writer; Modal commits the Volume in the
# background every few seconds and on shutdown. Move to Postgres before scaling out.
@app.function(image=image, volumes={"/data": db_volume}, secrets=[modal.Secret.from_name("six-eyes-x")],
              cpu=2, memory=8192, max_containers=1, scaledown_window=600, timeout=600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import recon
    from analyze import nsfw_model, probe
    from main import app as api, face_model

    # Load every model when the container starts, not on the first enroll or scan.
    face_model()
    probe()
    nsfw_model()
    recon.models()
    return api
