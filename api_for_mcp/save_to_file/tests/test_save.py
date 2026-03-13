import base64


class TestRootEndpoint:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_no_auth_required(self, client):
        assert client.get("/").status_code == 200

    def test_response_structure(self, client):
        data = client.get("/").json()
        assert "service" in data
        assert "endpoints" in data

    def test_service_name(self, client):
        assert client.get("/").json()["service"] == "Save to File API"


class TestAuth:
    def test_missing_key_returns_401(self, client, simple_payload):
        assert client.post("/save", json=simple_payload).status_code == 401

    def test_wrong_key_returns_401(self, client, invalid_headers, simple_payload):
        assert client.post("/save", json=simple_payload, headers=invalid_headers).status_code == 401

    def test_401_detail_message(self, client, invalid_headers, simple_payload):
        response = client.post("/save", json=simple_payload, headers=invalid_headers)
        assert response.json()["detail"] == "Неверный или отсутствующий токен"


class TestSaveFile:
    def test_returns_200(self, client, valid_headers, simple_payload):
        assert client.post("/save", json=simple_payload, headers=valid_headers).status_code == 200

    def test_response_has_saved_path(self, client, valid_headers, simple_payload):
        data = client.post("/save", json=simple_payload, headers=valid_headers).json()
        assert "saved_path" in data
        assert data["saved_path"]

    def test_response_has_filename(self, client, valid_headers, simple_payload):
        data = client.post("/save", json=simple_payload, headers=valid_headers).json()
        assert "filename" in data
        assert data["filename"] == simple_payload["filename"]

    def test_response_has_size_bytes(self, client, valid_headers, simple_payload):
        data = client.post("/save", json=simple_payload, headers=valid_headers).json()
        assert "size_bytes" in data
        assert data["size_bytes"] > 0

    def test_size_bytes_matches_decoded_length(self, client, valid_headers, simple_payload):
        data = client.post("/save", json=simple_payload, headers=valid_headers).json()
        assert data["size_bytes"] == len(base64.b64decode(simple_payload["content_base64"]))

    def test_file_exists_on_disk(self, client, valid_headers, simple_payload, save_dir):
        data = client.post("/save", json=simple_payload, headers=valid_headers).json()
        from pathlib import Path
        assert Path(data["saved_path"]).exists()

    def test_file_content_matches(self, client, valid_headers, simple_payload, save_dir):
        from pathlib import Path
        data = client.post("/save", json=simple_payload, headers=valid_headers).json()
        content = Path(data["saved_path"]).read_bytes()
        assert content == base64.b64decode(simple_payload["content_base64"])

    def test_with_subfolder(self, client, valid_headers, save_dir):
        from pathlib import Path
        payload = {
            "filename": "sub.txt",
            "content_base64": base64.b64encode(b"data").decode(),
            "subfolder": "mysubdir",
        }
        data = client.post("/save", json=payload, headers=valid_headers).json()
        p = Path(data["saved_path"])
        assert p.exists()
        assert p.parent.name == "mysubdir"

    def test_without_subfolder(self, client, valid_headers, save_dir):
        from pathlib import Path
        payload = {
            "filename": "nosub.txt",
            "content_base64": base64.b64encode(b"nosub").decode(),
        }
        data = client.post("/save", json=payload, headers=valid_headers).json()
        assert Path(data["saved_path"]).exists()


class TestValidation:
    def test_empty_filename_returns_422(self, client, valid_headers):
        payload = {"filename": "", "content_base64": base64.b64encode(b"x").decode()}
        assert client.post("/save", json=payload, headers=valid_headers).status_code == 422

    def test_empty_content_base64_returns_422(self, client, valid_headers):
        payload = {"filename": "file.txt", "content_base64": ""}
        assert client.post("/save", json=payload, headers=valid_headers).status_code == 422

    def test_invalid_base64_returns_422(self, client, valid_headers):
        payload = {"filename": "file.txt", "content_base64": "not-valid-base64!!!"}
        assert client.post("/save", json=payload, headers=valid_headers).status_code == 422

    def test_missing_filename_returns_422(self, client, valid_headers):
        payload = {"content_base64": base64.b64encode(b"x").decode()}
        assert client.post("/save", json=payload, headers=valid_headers).status_code == 422

    def test_missing_content_returns_422(self, client, valid_headers):
        payload = {"filename": "file.txt"}
        assert client.post("/save", json=payload, headers=valid_headers).status_code == 422


class TestDefaultSaveDir:
    """Regression: without SAVE_DIR env var files must go to <module>/saved_files/, not cwd."""

    def test_default_dir_is_module_saved_files_not_cwd(self, client, valid_headers, simple_payload, monkeypatch, tmp_path):
        # Remove SAVE_DIR so main.py falls back to Path(__file__).parent / "saved_files"
        monkeypatch.delenv("SAVE_DIR", raising=False)
        # Change cwd to tmp_path to prove we don't rely on cwd
        monkeypatch.chdir(tmp_path)
        import main as m
        expected_dir = (m.__file__ and str((m.__spec__.origin and __import__("pathlib").Path(m.__spec__.origin).parent / "saved_files")))
        data = client.post("/save", json=simple_payload, headers=valid_headers).json()
        saved = __import__("pathlib").Path(data["saved_path"])
        assert saved.exists()
        # Must NOT be inside cwd (tmp_path)
        assert not str(saved).startswith(str(tmp_path)), (
            f"File was saved under cwd {tmp_path} instead of module saved_files dir"
        )

    def test_default_dir_contains_saved_files_segment(self, client, valid_headers, simple_payload, monkeypatch, tmp_path):
        monkeypatch.delenv("SAVE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        data = client.post("/save", json=simple_payload, headers=valid_headers).json()
        saved = __import__("pathlib").Path(data["saved_path"])
        assert "saved_files" in saved.parts, (
            f"Expected 'saved_files' in path, got: {saved}"
        )

    def test_save_dir_env_var_overrides_default(self, client, valid_headers, simple_payload, monkeypatch, tmp_path):
        """SAVE_DIR env var must override, but must not be set to '.' (relative)."""
        explicit_dir = tmp_path / "explicit"
        monkeypatch.setenv("SAVE_DIR", str(explicit_dir))
        data = client.post("/save", json=simple_payload, headers=valid_headers).json()
        saved = __import__("pathlib").Path(data["saved_path"])
        assert str(saved).startswith(str(explicit_dir))

    def test_relative_save_dir_dot_is_dangerous(self, monkeypatch, tmp_path):
        """SAVE_DIR='.' resolves to cwd — this is the bug that caused /tmp/saved_files."""
        import os
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SAVE_DIR", ".")
        from pathlib import Path
        resolved = Path(os.getenv("SAVE_DIR")).resolve()
        # When cwd is /tmp, SAVE_DIR=. resolves to /tmp — NOT the module dir
        assert resolved == tmp_path.resolve(), "Confirms SAVE_DIR='.' depends on cwd"
        # Therefore .env must NOT contain SAVE_DIR=.
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            content = env_file.read_text()
            assert "SAVE_DIR=." not in content, (
                ".env must not set SAVE_DIR=. — it makes save location depend on cwd"
            )


class TestPathTraversal:
    def _b64(self):
        return base64.b64encode(b"x").decode()

    def test_filename_traversal_dotdot(self, client, valid_headers):
        payload = {"filename": "../../etc/passwd", "content_base64": self._b64()}
        assert client.post("/save", json=payload, headers=valid_headers).status_code == 422

    def test_filename_absolute_path(self, client, valid_headers):
        payload = {"filename": "/etc/passwd", "content_base64": self._b64()}
        assert client.post("/save", json=payload, headers=valid_headers).status_code == 422

    def test_filename_windows_traversal(self, client, valid_headers):
        payload = {"filename": "..\\secret", "content_base64": self._b64()}
        assert client.post("/save", json=payload, headers=valid_headers).status_code == 422

    def test_subfolder_traversal(self, client, valid_headers):
        payload = {"filename": "file.txt", "content_base64": self._b64(), "subfolder": "../.."}
        assert client.post("/save", json=payload, headers=valid_headers).status_code == 422

    def test_subfolder_nested_path(self, client, valid_headers):
        payload = {"filename": "file.txt", "content_base64": self._b64(), "subfolder": "a/b"}
        assert client.post("/save", json=payload, headers=valid_headers).status_code == 422
