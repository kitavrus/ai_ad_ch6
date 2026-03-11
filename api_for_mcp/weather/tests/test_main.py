class TestRootEndpoint:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_no_auth_required(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/").json()
        assert "service" in data
        assert "cities_count" in data
        assert "endpoints" in data

    def test_service_name(self, client):
        data = client.get("/").json()
        assert data["service"] == "Weather API"

    def test_cities_count_is_35(self, client):
        data = client.get("/").json()
        assert data["cities_count"] == 35

    def test_endpoints_list(self, client):
        data = client.get("/").json()
        assert "/weather/{city_name}" in data["endpoints"]
        assert "/cities" in data["endpoints"]


class TestCitiesEndpoint:
    def test_valid_key_returns_200(self, client, valid_headers):
        response = client.get("/cities", headers=valid_headers)
        assert response.status_code == 200

    def test_missing_key_returns_401(self, client):
        response = client.get("/cities")
        assert response.status_code == 401

    def test_invalid_key_returns_401(self, client, invalid_headers):
        response = client.get("/cities", headers=invalid_headers)
        assert response.status_code == 401

    def test_401_error_detail(self, client, invalid_headers):
        response = client.get("/cities", headers=invalid_headers)
        assert response.json()["detail"] == "Неверный или отсутствующий токен"

    def test_response_is_list(self, client, valid_headers):
        response = client.get("/cities", headers=valid_headers)
        assert isinstance(response.json(), list)

    def test_returns_35_cities(self, client, valid_headers):
        response = client.get("/cities", headers=valid_headers)
        assert len(response.json()) == 35

    def test_all_cities_are_strings(self, client, valid_headers):
        cities = client.get("/cities", headers=valid_headers).json()
        assert all(isinstance(c, str) for c in cities)

    def test_moskva_in_list(self, client, valid_headers):
        cities = client.get("/cities", headers=valid_headers).json()
        assert "Москва" in cities

    def test_spb_in_list(self, client, valid_headers):
        cities = client.get("/cities", headers=valid_headers).json()
        assert "Санкт-Петербург" in cities


class TestWeatherEndpoint:
    # Auth tests
    def test_missing_key_returns_401(self, client):
        response = client.get("/weather/москва")
        assert response.status_code == 401

    def test_invalid_key_returns_401(self, client, invalid_headers):
        response = client.get("/weather/москва", headers=invalid_headers)
        assert response.status_code == 401

    def test_401_detail_message(self, client, invalid_headers):
        response = client.get("/weather/москва", headers=invalid_headers)
        assert response.json()["detail"] == "Неверный или отсутствующий токен"

    # Direct Russian city lookup
    def test_russian_city_lowercase(self, client, valid_headers):
        response = client.get("/weather/москва", headers=valid_headers)
        assert response.status_code == 200
        assert response.json()["city"] == "Москва"

    def test_russian_city_uppercase(self, client, valid_headers):
        response = client.get("/weather/МОСКВА", headers=valid_headers)
        assert response.status_code == 200

    def test_russian_city_titlecase(self, client, valid_headers):
        response = client.get("/weather/Москва", headers=valid_headers)
        assert response.status_code == 200

    def test_russian_city_with_hyphen(self, client, valid_headers):
        response = client.get("/weather/ростов-на-дону", headers=valid_headers)
        assert response.status_code == 200

    def test_russian_multiword_city(self, client, valid_headers):
        response = client.get("/weather/нижний новгород", headers=valid_headers)
        assert response.status_code == 200

    # English aliases
    def test_english_alias_moscow(self, client, valid_headers):
        response = client.get("/weather/moscow", headers=valid_headers)
        assert response.status_code == 200
        assert response.json()["city"] == "Москва"

    def test_english_alias_spb(self, client, valid_headers):
        response = client.get("/weather/spb", headers=valid_headers)
        assert response.status_code == 200
        assert response.json()["city"] == "Санкт-Петербург"

    def test_english_alias_novosibirsk(self, client, valid_headers):
        response = client.get("/weather/novosibirsk", headers=valid_headers)
        assert response.status_code == 200

    def test_english_alias_saint_petersburg_hyphen(self, client, valid_headers):
        response = client.get("/weather/saint-petersburg", headers=valid_headers)
        assert response.status_code == 200

    def test_english_alias_yekaterinburg(self, client, valid_headers):
        response = client.get("/weather/yekaterinburg", headers=valid_headers)
        assert response.status_code == 200

    # Russian short aliases
    def test_russian_alias_msk(self, client, valid_headers):
        response = client.get("/weather/мск", headers=valid_headers)
        assert response.status_code == 200
        assert response.json()["city"] == "Москва"

    def test_russian_alias_piter(self, client, valid_headers):
        response = client.get("/weather/питер", headers=valid_headers)
        assert response.status_code == 200
        assert response.json()["city"] == "Санкт-Петербург"

    def test_russian_alias_nn(self, client, valid_headers):
        response = client.get("/weather/нн", headers=valid_headers)
        assert response.status_code == 200
        assert response.json()["city"] == "Нижний Новгород"

    def test_russian_alias_rostov(self, client, valid_headers):
        response = client.get("/weather/ростов", headers=valid_headers)
        assert response.status_code == 200
        assert response.json()["city"] == "Ростов-на-Дону"

    # Case-insensitive aliases
    def test_alias_case_insensitive_moscow_upper(self, client, valid_headers):
        response = client.get("/weather/MOSCOW", headers=valid_headers)
        assert response.status_code == 200

    def test_alias_case_insensitive_msk_upper(self, client, valid_headers):
        response = client.get("/weather/МСК", headers=valid_headers)
        assert response.status_code == 200

    # Response schema
    def test_response_has_all_fields(self, client, valid_headers):
        data = client.get("/weather/москва", headers=valid_headers).json()
        for field in ("city", "temperature", "condition", "humidity", "wind_speed", "wind_direction"):
            assert field in data

    def test_temperature_is_int(self, client, valid_headers):
        data = client.get("/weather/москва", headers=valid_headers).json()
        assert isinstance(data["temperature"], int)

    def test_humidity_range(self, client, valid_headers):
        data = client.get("/weather/москва", headers=valid_headers).json()
        assert 0 <= data["humidity"] <= 100

    def test_wind_speed_non_negative(self, client, valid_headers):
        data = client.get("/weather/москва", headers=valid_headers).json()
        assert data["wind_speed"] >= 0

    def test_city_field_matches_canonical(self, client, valid_headers):
        data = client.get("/weather/moscow", headers=valid_headers).json()
        assert data["city"] == "Москва"

    # Not found
    def test_unknown_city_returns_404(self, client, valid_headers):
        response = client.get("/weather/london", headers=valid_headers)
        assert response.status_code == 404

    def test_unknown_city_error_detail(self, client, valid_headers):
        response = client.get("/weather/london", headers=valid_headers)
        detail = response.json()["detail"]
        assert "london" in detail
        assert "Доступные города" in detail

    def test_completely_bogus_name(self, client, valid_headers):
        response = client.get("/weather/zzzzz", headers=valid_headers)
        assert response.status_code == 404
