"""
dvc_backend.py
──────────────
Backend genérico DVC: usa dvc.api.open() para leer un archivo trackeado por DVC
desde cualquier remote configurado (S3, GCS, SSH, local, etc.).

No requiere que el repositorio DVC esté completamente inicializado en el
contenedor; dvc.api puede trabajar con la URL del remote directamente.
El archivo se descarga a un temporal y se devuelve su ruta local.
"""

import tempfile
from pathlib import Path
from typing import Optional

from .base import StorageBackend


class DvcBackend(StorageBackend):
    """Accede a datos vía DVC API (compatible con S3, GCS, SSH, local DVC remote, etc.)

    Atributos configurados desde StorageConfig (YAML):
      remote   : nombre del remote en .dvc/config (None → DVC usa el default)
      rev      : tag/commit/branch de git para reproducibilidad (None → HEAD)
      auto_pull: no usado directamente aquí (dvc.api siempre hace pull si hace falta)
    """

    def __init__(self, cfg):
        self.remote: Optional[str] = getattr(cfg, "remote", None)
        self.rev: Optional[str] = getattr(cfg, "rev", None)

    def resolve(self, path: str) -> Path:
        try:
            import dvc.api
        except ImportError as exc:
            raise ImportError(
                "DvcBackend requiere el paquete 'dvc'. "
                "Añade 'dvc' (y el plugin del remote, p.ej. dvc-s3) al Dockerfile."
            ) from exc

        kwargs: dict = {}
        if self.remote:
            kwargs["remote"] = self.remote
        if self.rev:
            kwargs["rev"] = self.rev

        suffix = Path(path).suffix or ".tmp"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            with dvc.api.open(path, mode="rb", **kwargs) as src:
                tmp.write(src.read())
        finally:
            tmp.close()

        return Path(tmp.name)
