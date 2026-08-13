import tomllib
from pathlib import Path


def test_libero_spatial_task_mapping_is_a_complete_bijection() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "libero_spatial_tasks.toml"
    )
    with path.open("rb") as source:
        mapping = tomllib.load(source)

    assert mapping["schema_version"] == 1
    assert mapping["benchmark_suite"] == "libero_spatial"
    tasks = mapping["tasks"]
    assert len(tasks) == 10
    assert {task["dataset_task_index"] for task in tasks} == set(range(10))
    assert {task["benchmark_task_id"] for task in tasks} == set(range(10))
    assert len({task["language"] for task in tasks}) == 10
