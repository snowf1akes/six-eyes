"""Serve the Six Eyes API on Modal at a stable URL.

Deploy from the repo root, on a machine that has the trained heads in data/models (ml/heads.py):
    .venv/Scripts/modal deploy api/modal_app.py
URL: https://<workspace>--six-eyes-api-web.modal.run
Needs three Modal secrets: six-eyes-x (X_BEARER_TOKEN, plus PHOTON_API_KEY, PHOTON_PROJECT_ID and
PHOTON_AGENT_USER_ID for texts), huggingface (HF_TOKEN, for the gated DINOv3) and factory
(FACTORY_API_KEY, for the case worker). The case worker owns the 48-hour clock; opening the
dashboard or the hourly deadline_sweep only wakes overdue Factory sessions if that timer died.
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
    .apt_install("libgl1", "libglib2.0-0", "curl")  # OpenCV's runtime libraries; curl for the droid installer
    .run_commands("curl -fsSL https://app.factory.ai/cli | sh")  # the droid CLI, for the case worker
    .pip_install_from_requirements(str(Path(__file__).with_name("requirements.txt")))
    # Bake the face, DINOv3, NSFW, VAE and LPIPS weights into the image; the token is only needed for this step.
    .run_function(fetch_models, secrets=[modal.Secret.from_name("huggingface")])
    .env({"DB_PATH": "/data/six_eyes.db", "MODELS_DIR": "/root/models", "HF_HUB_OFFLINE": "1",
          "DROID_PATH": "/root/.local/bin/droid",
          "PHOTON_PLATFORM": "imessage",
          "APP_URL": "https://six-eyes-henna.vercel.app"})
    .add_local_python_source("main", "x", "signals", "analyze", "face", "crops", "recon", "channels",
                            "caseworker", "photon")
    .add_local_file(Path(__file__).with_name("data") / "keywords.json", "/root/data/keywords.json")
    .add_local_file(Path(__file__).with_name("data") / "channels.json", "/root/data/channels.json")
    .add_local_file(Path(__file__).with_name("templates") / "notice.md", "/root/templates/notice.md")
    .add_local_file(Path(__file__).with_name("templates") / "details.md", "/root/templates/details.md")
    .add_local_file(Path(__file__).with_name("templates") / "ftc_packet.md", "/root/templates/ftc_packet.md")
    .add_local_file(MODELS / "dinov3_binary.joblib", "/root/models/dinov3_binary.joblib")
    .add_local_file(MODELS / "dinov3_multi.joblib", "/root/models/dinov3_multi.joblib")
)


# ponytail: one container is the only SQLite writer; Modal commits the Volume in the
# background every few seconds and on shutdown. Move to Postgres before scaling out.
@app.function(image=image, volumes={"/data": db_volume},
              secrets=[modal.Secret.from_name("six-eyes-x"), modal.Secret.from_name("factory")],
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


# Calls the web app rather than opening the database, so the web container stays the only SQLite writer.
@app.function(image=modal.Image.debian_slim(python_version="3.12").pip_install("httpx"),
              schedule=modal.Period(hours=1))
def deadline_sweep():
    import httpx

    r = httpx.post(web.get_web_url() + "/recheck", timeout=300)
    r.raise_for_status()
    print(r.json())
