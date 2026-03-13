import base64
import os
import uvicorn
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator

load_dotenv()

app = FastAPI(title="Save to File API", description="File saving service")

API_KEY = os.getenv("SAVE_API_KEY", "secret-token")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_token(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий токен")


class SaveFileRequest(BaseModel):
    filename: str
    content_base64: str
    subfolder: Optional[str] = None

    @field_validator("filename")
    @classmethod
    def filename_safe(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("filename не может быть пустым")
        if "\\" in v or Path(v).name != v:
            raise ValueError("filename не должен содержать разделители пути")
        return v

    @field_validator("content_base64")
    @classmethod
    def content_valid_base64(cls, v: str) -> str:
        if not v:
            raise ValueError("content_base64 не может быть пустым")
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("content_base64 содержит невалидный base64")
        return v

    @field_validator("subfolder")
    @classmethod
    def subfolder_safe(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if Path(v).name != v:
            raise ValueError("subfolder не должен содержать разделители пути")
        return v


class SaveFileResponse(BaseModel):
    saved_path: str
    filename: str
    size_bytes: int


@app.get("/")
def root():
    return {"service": "Save to File API", "endpoints": ["/save"]}


@app.post("/save", response_model=SaveFileResponse, dependencies=[Depends(verify_token)])
def save_file(body: SaveFileRequest):
    _default_dir = Path(__file__).resolve().parent / "saved_files"
    save_dir = Path(os.getenv("SAVE_DIR", str(_default_dir))).resolve()

    target_dir = save_dir / body.subfolder if body.subfolder else save_dir
    target_file = target_dir / body.filename

    if not str(target_file.resolve()).startswith(str(save_dir)):
        raise HTTPException(status_code=400, detail="Недопустимый путь к файлу")

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        file_bytes = base64.b64decode(body.content_base64)
        target_file.write_bytes(file_bytes)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка записи файла: {e.strerror}")

    return SaveFileResponse(
        saved_path=str(target_file.resolve()),
        filename=body.filename,
        size_bytes=len(file_bytes),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8884)
