import sys

sys.path.append("../..")
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from compressor import pipeline
from kmeans import KMEANS_DEFAULTS

app = FastAPI(
    title="K-means Compressor",
    description="Compressing jpg/png images using k-means clustering algorithm implemented from scratch.",
)

app.frontend("/", directory="../frontend")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "K-means Compressor works!"}


@app.post("/compress/")
async def run_pipeline(
    file: UploadFile,
    k: int = Form(gt=1, le=256),
    seed: int = Form(default=KMEANS_DEFAULTS["seed"]),
    max_iter: int = Form(default=KMEANS_DEFAULTS["max_iter"]),
    eps: float = Form(default=KMEANS_DEFAULTS["eps"]),
    batch_size: int = Form(default=KMEANS_DEFAULTS["batch_size"]),
    backend: Literal["scratch", "sklearn"] = Form(default="scratch"),
):
    input_path: Path = Path("../uploads") / file.filename
    input_path.parent.mkdir(parents=True, exist_ok=True)

    if not file.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        return HTMLResponse(
            status_code=400, content="Compressor supports only .jpg or .png files!"
        )

    contents = await file.read()
    input_path.write_bytes(contents)

    output_path = Path("../results") / f"compressed_{file.filename}"
    pipeline(
        load_path=input_path,
        save_path=output_path,
        k=k,
        seed=seed,
        max_iter=max_iter,
        eps=eps,
        batch_size=batch_size,
        backend=backend,
    )

    return FileResponse(output_path)
