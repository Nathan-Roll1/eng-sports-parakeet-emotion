from __future__ import annotations

from fastapi import FastAPI

from .iu_annotator_app import mount_iu_annotator


app = FastAPI(title="IU Emotion Annotator")
mount_iu_annotator(app)
