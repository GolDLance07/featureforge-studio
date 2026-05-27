from io import BytesIO

from core.dataset_manager import DatasetManager


def test_load_csv_deduplicates_columns():
    csv = BytesIO(b"age,age,city\n10,20,Pune\n")
    csv.name = "sample.csv"

    result = DatasetManager().load_csv(csv)

    assert result.filename == "sample.csv"
    assert result.dataframe.columns.tolist() == ["age", "age_1", "city"]
