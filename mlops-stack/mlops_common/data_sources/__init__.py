from .base import DataSource, DatasetBundle
from .csv_source import CsvDataSource
from .sklearn_dataset import SklearnDatasetSource
from .hri_csv import HriCsvDataSource

DATA_SOURCE_REGISTRY = {
    "sklearn_dataset": SklearnDatasetSource,
    "csv":             CsvDataSource,
    "hri_csv":         HriCsvDataSource,
}


def get_data_source(cfg) -> DataSource:
    """cfg: DataSourceConfig (mlops_common.config)"""
    try:
        cls = DATA_SOURCE_REGISTRY[cfg.type]
    except KeyError:
        raise ValueError(f"Unknown data source type: {cfg.type!r}")
    return cls(cfg)


__all__ = ["DataSource", "DatasetBundle", "get_data_source", "DATA_SOURCE_REGISTRY"]
