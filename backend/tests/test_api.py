from pathlib import Path


def _csv(seed_dir: Path, name: str) -> bytes:
    return (seed_dir / name).read_bytes()


def test_health_reports_database_and_counts(client):
    body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["datasets"] == 7


def test_catalog_is_bilingual(client):
    datasets = client.get("/api/datasets").json()["datasets"]

    assert len(datasets) == 7
    for item in datasets:
        assert item["title"]["en"] and item["title"]["ru"]
        assert item["why"]["en"] and item["why"]["ru"]
        assert "csv" not in item


def test_single_dataset_includes_csv(client):
    body = client.get("/api/datasets/students").json()

    assert body["slug"] == "students"
    assert body["target"] == "exam_score"
    assert body["csv"].startswith("hours_studied")


def test_unknown_dataset_returns_404(client):
    assert client.get("/api/datasets/nope").status_code == 404


def test_analyze_returns_best_feature(client, seed_dir: Path):
    response = client.post(
        "/api/analyze",
        files={"file": ("students.csv", _csv(seed_dir, "students.csv"), "text/csv")},
        data={"target": "exam_score"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["best_feature"] == "hours_studied"
    assert body["task"] == "regression"


def test_analyze_rejects_unknown_target(client, seed_dir: Path):
    response = client.post(
        "/api/analyze",
        files={"file": ("students.csv", _csv(seed_dir, "students.csv"), "text/csv")},
        data={"target": "no_such_column"},
    )

    assert response.status_code == 400


def test_analyze_rejects_empty_file(client):
    response = client.post(
        "/api/analyze",
        files={"file": ("empty.csv", b"", "text/csv")},
        data={"target": "whatever"},
    )

    assert response.status_code == 400


def test_analyze_writes_history(client, seed_dir: Path):
    assert client.get("/api/history").json()["total"] == 0

    client.post(
        "/api/analyze",
        files={"file": ("students.csv", _csv(seed_dir, "students.csv"), "text/csv")},
        data={"target": "exam_score"},
    )

    history = client.get("/api/history").json()
    assert history["total"] == 1
    run = history["runs"][0]
    assert run["filename"] == "students.csv"
    assert run["target"] == "exam_score"
    assert run["best_feature"] == "hours_studied"
    assert run["rows"] == 200


def test_history_can_be_cleared(client, seed_dir: Path):
    client.post(
        "/api/analyze",
        files={"file": ("students.csv", _csv(seed_dir, "students.csv"), "text/csv")},
        data={"target": "exam_score"},
    )

    assert client.delete("/api/history").json()["deleted"] == 1
    assert client.get("/api/history").json()["total"] == 0


def test_predict_returns_value(client, seed_dir: Path):
    response = client.post(
        "/api/predict",
        files={"file": ("students.csv", _csv(seed_dir, "students.csv"), "text/csv")},
        data={
            "target": "exam_score",
            "values": '{"hours_studied": "8", "sleep_hours": "7", "attendance_pct": "90", "prev_score": "70"}',
        },
    )

    assert response.status_code == 200
    assert response.json()["target"] == "exam_score"
