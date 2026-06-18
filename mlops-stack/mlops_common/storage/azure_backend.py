"""
azure_backend.py
────────────────
Backend Azure Blob Storage: descarga un blob directamente usando el SDK
azure-storage-blob, sin pasar por DVC.

Credenciales (en orden de prioridad):
  1. cfg.connection_string  (hardcodeado en config, solo para dev local)
  2. Var de entorno AZURE_STORAGE_CONNECTION_STRING
  3. DefaultAzureCredential (Managed Identity, Azure CLI, env vars AZURE_CLIENT_*)

En producción se recomienda la opción 3 via Managed Identity o AZURE_CLIENT_*.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from .base import StorageBackend


class AzureBlobBackend(StorageBackend):
    """Descarga un blob de Azure Blob Storage y devuelve su ruta temporal local.

    YAML de configuración:
      storage:
        backend: azure
        container: mi-contenedor          # nombre del contenedor Azure
        connection_string: "..."          # opcional; preferir variable de entorno
    """

    def __init__(self, cfg):
        self.container: str = getattr(cfg, "container", "") or ""
        self._conn_str: Optional[str] = getattr(cfg, "connection_string", None)

    def resolve(self, path: str) -> Path:
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:
            raise ImportError(
                "AzureBlobBackend requiere 'azure-storage-blob'. "
                "Añade al Dockerfile: pip install azure-storage-blob"
            ) from exc

        conn_str = self._conn_str or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

        if conn_str:
            client = BlobServiceClient.from_connection_string(conn_str)
        else:
            # Managed Identity / Azure CLI / env vars AZURE_CLIENT_*
            try:
                from azure.identity import DefaultAzureCredential
                from azure.storage.blob import BlobServiceClient as _BSC
            except ImportError as exc:
                raise ImportError(
                    "Para autenticación sin connection string instala 'azure-identity'."
                ) from exc
            account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
            if not account_name:
                raise EnvironmentError(
                    "AzureBlobBackend: define AZURE_STORAGE_CONNECTION_STRING "
                    "o AZURE_STORAGE_ACCOUNT_NAME en el entorno."
                )
            url = f"https://{account_name}.blob.core.windows.net"
            client = _BSC(account_url=url, credential=DefaultAzureCredential())

        if not self.container:
            raise ValueError(
                "AzureBlobBackend: 'container' no definido en la config de storage."
            )

        blob_name = path.lstrip("/")
        suffix = Path(blob_name).suffix or ".tmp"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            blob_client = client.get_blob_client(container=self.container, blob=blob_name)
            tmp.write(blob_client.download_blob().readall())
        finally:
            tmp.close()

        return Path(tmp.name)
