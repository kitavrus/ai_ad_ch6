import base64

import pytest


class TestRootEndpoint:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_no_auth_required(self, client):
        assert client.get("/").status_code == 200

    def test_response_structure(self, client):
        data = client.get("/").json()
        assert "service" in data
        assert "endpoints" in data

    def test_service_name(self, client):
        assert client.get("/").json()["service"] == "PDF Maker API"


class TestAuth:
    def test_missing_key_returns_401(self, client, simple_payload):
        response = client.post("/pdf", json=simple_payload)
        assert response.status_code == 401

    def test_wrong_key_returns_401(self, client, invalid_headers, simple_payload):
        response = client.post("/pdf", json=simple_payload, headers=invalid_headers)
        assert response.status_code == 401

    def test_401_detail_message(self, client, invalid_headers, simple_payload):
        response = client.post("/pdf", json=simple_payload, headers=invalid_headers)
        assert response.json()["detail"] == "Неверный или отсутствующий токен"


class TestCreatePdf:
    def test_returns_200(self, client, valid_headers, simple_payload):
        response = client.post("/pdf", json=simple_payload, headers=valid_headers)
        assert response.status_code == 200

    def test_response_has_pdf_base64(self, client, valid_headers, simple_payload):
        data = client.post("/pdf", json=simple_payload, headers=valid_headers).json()
        assert "pdf_base64" in data
        assert data["pdf_base64"]

    def test_response_has_filename(self, client, valid_headers, simple_payload):
        data = client.post("/pdf", json=simple_payload, headers=valid_headers).json()
        assert "filename" in data
        assert data["filename"].endswith(".pdf")

    def test_response_has_size_bytes(self, client, valid_headers, simple_payload):
        data = client.post("/pdf", json=simple_payload, headers=valid_headers).json()
        assert "size_bytes" in data
        assert data["size_bytes"] > 0

    def test_base64_decodes_to_valid_pdf(self, client, valid_headers, simple_payload):
        data = client.post("/pdf", json=simple_payload, headers=valid_headers).json()
        pdf_bytes = base64.b64decode(data["pdf_base64"])
        assert pdf_bytes[:4] == b"%PDF"

    def test_size_bytes_matches_decoded_length(self, client, valid_headers, simple_payload):
        data = client.post("/pdf", json=simple_payload, headers=valid_headers).json()
        pdf_bytes = base64.b64decode(data["pdf_base64"])
        assert data["size_bytes"] == len(pdf_bytes)

    def test_filename_derived_from_title(self, client, valid_headers):
        payload = {"title": "My Report"}
        data = client.post("/pdf", json=payload, headers=valid_headers).json()
        assert "my_report" in data["filename"]

    def test_title_only_no_sections(self, client, valid_headers, simple_payload):
        response = client.post("/pdf", json=simple_payload, headers=valid_headers)
        assert response.status_code == 200

    def test_with_author(self, client, valid_headers):
        payload = {"title": "Report", "author": "Jane Doe"}
        response = client.post("/pdf", json=payload, headers=valid_headers)
        assert response.status_code == 200

    def test_with_heading_and_content(self, client, valid_headers):
        payload = {"title": "Doc", "sections": [{"heading": "Intro", "content": "Hello world."}]}
        response = client.post("/pdf", json=payload, headers=valid_headers)
        assert response.status_code == 200

    def test_with_items_list(self, client, valid_headers):
        payload = {"title": "Doc", "sections": [{"heading": "List", "items": ["A", "B", "C"]}]}
        response = client.post("/pdf", json=payload, headers=valid_headers)
        assert response.status_code == 200

    def test_with_table(self, client, valid_headers):
        payload = {
            "title": "Doc",
            "sections": [
                {
                    "heading": "Table",
                    "table": {
                        "headers": ["Col1", "Col2"],
                        "rows": [["r1c1", "r1c2"], ["r2c1", "r2c2"]],
                    },
                }
            ],
        }
        response = client.post("/pdf", json=payload, headers=valid_headers)
        assert response.status_code == 200

    def test_full_document(self, client, valid_headers, full_payload):
        data = client.post("/pdf", json=full_payload, headers=valid_headers).json()
        pdf_bytes = base64.b64decode(data["pdf_base64"])
        assert pdf_bytes[:4] == b"%PDF"

    def test_empty_title_returns_422(self, client, valid_headers):
        response = client.post("/pdf", json={"title": ""}, headers=valid_headers)
        assert response.status_code == 422

    def test_missing_title_returns_422(self, client, valid_headers):
        response = client.post("/pdf", json={"author": "No Title"}, headers=valid_headers)
        assert response.status_code == 422

    def test_empty_sections_list(self, client, valid_headers):
        payload = {"title": "No Sections", "sections": []}
        response = client.post("/pdf", json=payload, headers=valid_headers)
        assert response.status_code == 200
