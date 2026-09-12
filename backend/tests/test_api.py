from pathlib import Path

CATALOG_SIZE = 10


def _csv(seed_dir: Path, name: str) -> bytes:
    return (seed_dir / name).read_bytes()


def test_health_reports_database_and_counts(client):
    body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["datasets"] == CATALOG_SIZE


def test_catalog_is_bilingual(client):
    datasets = client.get("/api/datasets").json()["datasets"]

    assert len(datasets) == CATALOG_SIZE
    for item in datasets:
        assert item["title"]["en"] and item["title"]["ru"]
        assert item["why"]["en"] and item["why"]["ru"]
        assert "csv" not in item


def test_every_dataset_points_at_kaggle(client):
    datasets = client.get("/api/datasets").json()["datasets"]

    for item in datasets:
        assert item["source"]["url"].startswith("https://www.kaggle.com/datasets/")
        assert item["source"]["name"].startswith("Kaggle")


def test_single_dataset_includes_csv(client):
    body = client.get("/api/datasets/iris").json()

    assert body["slug"] == "iris"
    assert body["target"] == "Species"
    assert body["csv"].startswith("Id,SepalLengthCm")


def test_unknown_dataset_returns_404(client):
    assert client.get("/api/datasets/nope").status_code == 404


def test_analyze_returns_best_feature(client, seed_dir: Path):
    response = client.post(
        "/api/analyze",
        files={"file": ("iris.csv", _csv(seed_dir, "iris.csv"), "text/csv")},
        data={"target": "Species"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["best_feature"] == "PetalWidthCm"
    assert body["task"] == "classification"


def test_analyze_reports_what_it_cleaned_up(client, seed_dir: Path):
    response = client.post(
        "/api/analyze",
        files={"file": ("iris.csv", _csv(seed_dir, "iris.csv"), "text/csv")},
        data={"target": "Species"},
    )

    notes = response.json()["cleanup"]
    assert any(note["column"] == "Id" and note["action"] == "id_column" for note in notes)


def test_analyze_handles_raw_kaggle_columns(client, seed_dir: Path):
    response = client.post(
        "/api/analyze",
        files={"file": ("telco_churn.csv", _csv(seed_dir, "telco_churn.csv"), "text/csv")},
        data={"target": "Churn"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "TotalCharges" in body["feature_scores"]
    assert any(note["column"] == "customerID" for note in body["cleanup"])


def test_analyze_rejects_unknown_target(client, seed_dir: Path):
    response = client.post(
        "/api/analyze",
        files={"file": ("iris.csv", _csv(seed_dir, "iris.csv"), "text/csv")},
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
        files={"file": ("iris.csv", _csv(seed_dir, "iris.csv"), "text/csv")},
        data={"target": "Species"},
    )

    history = client.get("/api/history").json()
    assert history["total"] == 1
    run = history["runs"][0]
    assert run["filename"] == "iris.csv"
    assert run["target"] == "Species"
    assert run["best_feature"] == "PetalWidthCm"
    assert run["rows"] == 150


def test_history_can_be_cleared(client, seed_dir: Path):
    client.post(
        "/api/analyze",
        files={"file": ("iris.csv", _csv(seed_dir, "iris.csv"), "text/csv")},
        data={"target": "Species"},
    )

    assert client.delete("/api/history").json()["deleted"] == 1
    assert client.get("/api/history").json()["total"] == 0


def test_predict_returns_value(client, seed_dir: Path):
    response = client.post(
        "/api/predict",
        files={"file": ("iris.csv", _csv(seed_dir, "iris.csv"), "text/csv")},
        data={
            "target": "Species",
            "values": '{"SepalLengthCm": "5.1", "SepalWidthCm": "3.5", "PetalLengthCm": "1.4", "PetalWidthCm": "0.2"}',
        },
    )

    assert response.status_code == 200
    assert response.json()["target"] == "Species"


def test_seeder_removes_datasets_that_left_the_catalog(client):
    import db

    with db.Session(db.engine) as session:
        session.add(
            db.Dataset(
                slug="retired",
                title_en="Retired",
                title_ru="Удалённый",
                target="x",
                task="regression",
                source_name="",
                source_url="",
                why_en="gone",
                why_ru="удалён",
                rows=1,
                cols=1,
                csv="x\n1\n",
            )
        )
        session.commit()

    assert db.count_datasets() == CATALOG_SIZE + 1

    db.seed_datasets()

    assert db.count_datasets() == CATALOG_SIZE
    assert db.get_dataset("retired") is None
