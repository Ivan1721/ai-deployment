"""
gdrive_backend.py
─────────────────
Backend Google Drive: usa DVC con remote gdrive:// para descargar el archivo.
DVC gestiona el flujo OAuth y el caché local; este módulo simplemente delega
en DvcBackend con el remote name apropiado.

Configuración del remote DVC (una sola vez, fuera del contenedor):
  dvc remote add gdrive gdrive://<folder_id>

Para entornos CI/CD o Docker (sin interacción):
  dvc remote modify gdrive gdrive_use_service_account true
  dvc remote modify gdrive gdrive_service_account_json_file_path /run/secrets/gdrive-sa.json

El archivo de Service Account debe montarse como secreto Docker o variable de entorno
GDRIVE_CREDENTIALS_DATA (JSON del SA encodificado en base64).
"""

from pathlib import Path
from typing import Optional

from .base import StorageBackend


class GoogleDriveBackend(StorageBackend):
    """Accede a datos en Google Drive vía DVC remote gdrive://.

    YAML de configuración:
      storage:
        backend: gdrive
        remote: gdrive          # nombre del remote en .dvc/config
        rev: v1.0               # opcional: revisión git para reproducibilidad
    """

    def __init__(self, cfg):
        self.remote: Optional[str] = getattr(cfg, "remote", None) or "gdrive"
        self.rev: Optional[str] = getattr(cfg, "rev", None)

    def resolve(self, path: str) -> Path:
        from .dvc_backend import DvcBackend

        class _Cfg:
            remote = self.remote
            rev = self.rev

        return DvcBackend(_Cfg()).resolve(path)
