from .base import StorageBackend
from .git_local import GitLocalBackend
from .dvc_backend import DvcBackend
from .azure_backend import AzureBlobBackend
from .gdrive_backend import GoogleDriveBackend
from .encrypted_local_backend import EncryptedLocalBackend

STORAGE_REGISTRY: dict = {
    "git_local":       GitLocalBackend,
    "dvc":             DvcBackend,
    "azure":           AzureBlobBackend,
    "gdrive":          GoogleDriveBackend,
    "encrypted_local": EncryptedLocalBackend,
}


def get_storage_backend(cfg) -> StorageBackend:
    """cfg: StorageConfig (mlops_common.config). Devuelve la instancia del backend."""
    backend_type = getattr(cfg, "backend", "git_local")
    try:
        cls = STORAGE_REGISTRY[backend_type]
    except KeyError:
        raise ValueError(
            f"Backend de almacenamiento desconocido: {backend_type!r}. "
            f"Disponibles: {list(STORAGE_REGISTRY)}"
        )
    return cls(cfg)


__all__ = [
    "StorageBackend",
    "GitLocalBackend",
    "DvcBackend",
    "AzureBlobBackend",
    "GoogleDriveBackend",
    "EncryptedLocalBackend",
    "STORAGE_REGISTRY",
    "get_storage_backend",
]
