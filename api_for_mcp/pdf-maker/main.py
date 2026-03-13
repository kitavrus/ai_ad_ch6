import base64
import os
import re
import uvicorn
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from fpdf import FPDF
from pydantic import BaseModel, field_validator

load_dotenv()

app = FastAPI(title="PDF Maker API", description="PDF generation service")

API_KEY = os.getenv("PDF_API_KEY", "secret-token")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

FONTS_DIR = Path(__file__).parent / "fonts"
DEJAVU_FONT = FONTS_DIR / "DejaVuSans.ttf"
DEJAVU_BOLD_FONT = FONTS_DIR / "DejaVuSans-Bold.ttf"


async def verify_token(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий токен")


class TableData(BaseModel):
    headers: List[str]
    rows: List[List[str]]


class Section(BaseModel):
    heading: Optional[str] = None
    content: Optional[str] = None
    items: Optional[List[str]] = None
    table: Optional[TableData] = None


class PdfRequest(BaseModel):
    title: str
    author: Optional[str] = None
    sections: List[Section] = []

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("title не может быть пустым")
        return v


class PdfResponse(BaseModel):
    pdf_base64: str
    filename: str
    size_bytes: int


def _slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text or "document"


def _build_pdf(req: PdfRequest) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    use_unicode = DEJAVU_FONT.exists()
    if use_unicode:
        pdf.add_font("DejaVu", "", str(DEJAVU_FONT))
        if DEJAVU_BOLD_FONT.exists():
            pdf.add_font("DejaVu", "B", str(DEJAVU_BOLD_FONT))
            font_name = "DejaVu"
        else:
            font_name = "DejaVu"
    else:
        font_name = "Helvetica"

    # Title page
    pdf.add_page()
    pdf.set_font(font_name, "B" if use_unicode and DEJAVU_BOLD_FONT.exists() else "", 24)
    pdf.cell(0, 20, req.title, new_x="LMARGIN", new_y="NEXT", align="C")

    if req.author:
        pdf.set_font(font_name, "", 14)
        pdf.cell(0, 10, req.author, new_x="LMARGIN", new_y="NEXT", align="C")

    # Sections
    for section in req.sections:
        pdf.ln(5)

        if section.heading:
            pdf.set_font(font_name, "B" if use_unicode and DEJAVU_BOLD_FONT.exists() else "", 16)
            pdf.cell(0, 10, section.heading, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        if section.content:
            pdf.set_font(font_name, "", 12)
            pdf.multi_cell(0, 7, section.content, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        if section.items:
            pdf.set_font(font_name, "", 12)
            for item in section.items:
                pdf.multi_cell(0, 7, f"- {item}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        if section.table:
            _render_table(pdf, section.table, font_name, use_unicode)

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def _render_table(pdf: FPDF, table: TableData, font_name: str, use_unicode: bool) -> None:
    col_count = len(table.headers)
    if col_count == 0:
        return
    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    col_width = page_width / col_count

    # Header row
    pdf.set_font(font_name, "B" if use_unicode and DEJAVU_BOLD_FONT.exists() else "", 11)
    for header in table.headers:
        pdf.cell(col_width, 8, header, border=1)
    pdf.ln()

    # Data rows
    pdf.set_font(font_name, "", 11)
    for row in table.rows:
        for i, cell in enumerate(row):
            value = cell if i < len(row) else ""
            pdf.cell(col_width, 7, value, border=1)
        pdf.ln()
    pdf.ln(3)


@app.get("/")
def root():
    return {"service": "PDF Maker API", "endpoints": ["/pdf"]}


@app.post("/pdf", response_model=PdfResponse, dependencies=[Depends(verify_token)])
def create_pdf(body: PdfRequest):
    pdf_bytes = _build_pdf(body)
    filename = f"{_slug(body.title)}.pdf"
    return PdfResponse(
        pdf_base64=base64.b64encode(pdf_bytes).decode("utf-8"),
        filename=filename,
        size_bytes=len(pdf_bytes),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8883)
