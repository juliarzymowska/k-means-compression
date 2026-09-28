from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from elbow import elbow_full
from src.compressor import pipeline
from src.kmeans import KMEANS_DEFAULTS

INPUT_PATH = Path("results/uploads")
OUTPUT_PATH_COMPRESSED = Path("results/compressed")
OUTPUT_PATH_ELBOW = Path("results/elbow_method")

app = FastAPI(
    title="K-means Compressor",
    description="Compressing jpg/png images using k-means clustering algorithm implemented from scratch.",
)

app.frontend("/", directory="src/web/frontend")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "K-means Compressor works!"}


@app.post("/compress/")
async def run_pipeline(
    file: UploadFile,
    k: int = Form(ge=1, le=256),
    seed: int = Form(default=KMEANS_DEFAULTS["seed"]),
    max_iter: int = Form(default=KMEANS_DEFAULTS["max_iter"]),
    eps: float = Form(default=KMEANS_DEFAULTS["eps"]),
    batch_size: int = Form(default=KMEANS_DEFAULTS["batch_size"]),
    backend: Literal["scratch", "sklearn"] = Form(default="scratch"),
):
    input_path: Path = INPUT_PATH / file.filename
    input_path.parent.mkdir(parents=True, exist_ok=True)

    if not file.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        return HTMLResponse(
            status_code=400, content="Compressor supports only .jpg or .png files!"
        )

    contents = await file.read()
    input_path.write_bytes(contents)

    output_path = OUTPUT_PATH_COMPRESSED / f"compressed_{file.filename}"
    report = pipeline(
        load_path=input_path,
        save_path=output_path,
        k=k,
        seed=seed,
        max_iter=max_iter,
        eps=eps,
        batch_size=batch_size,
        backend=backend,
    )

    # pipeline may change the suffix (always saves as .png)
    return FileResponse(report["save_path"])


@app.post("/elbow/")
async def run_elbow_method(
    file: UploadFile,
    max_k: int = Form(gt=2, le=256),
    seed: int = Form(default=KMEANS_DEFAULTS["seed"]),
):
    input_path: Path = INPUT_PATH / file.filename
    input_path.parent.mkdir(parents=True, exist_ok=True)

    if not file.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        return HTMLResponse(
            status_code=400, content="Compressor supports only .jpg or .png files!"
        )

    contents = await file.read()
    input_path.write_bytes(contents)

    output_path = OUTPUT_PATH_ELBOW / f"elbow_{file.filename}"

    elbow_full(max_k=max_k, load=input_path, seed=seed, save_path=output_path)

    return FileResponse(output_path)
