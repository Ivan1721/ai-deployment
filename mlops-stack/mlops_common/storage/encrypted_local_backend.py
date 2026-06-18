"""
encrypted_local_backend.py
──────────────────────────
Backend para almacenamiento local encriptado: el remote DVC apunta a una
carpeta en un volumen encriptado (VeraCrypt, LUKS, macOS FileVault, etc.)

Responsabilidades de este backend:
  1. Verificar que el volumen encriptado esté montado en encrypted_path.
  2. Delegar la resolución del archivo en DvcBackend usando el remote local.

Responsabilidad del operador (fuera del stack):
  - Crear y montar el volumen encriptado ANTES de `docker compose up`.
  - Configurar el remote DVC apuntando a ese volumen:
      dvc remote add local_enc /mnt/secure-volume/dvc-cache
  - En Docker: montar el volumen desencriptado como bind mount:
      volumes:
        - /mnt/secure-volume:/mnt/secure-volume:ro

El stack no gestiona el cifrado en sí; solo verifica que la ruta esté disponible.
"""

from pathlib import Path
from typing import Optional

from .base import StorageBackend


class EncryptedLocalBackend(StorageBackend):
    """Accede a datos en una carpeta local encriptada vía DVC remote local.

    YAML de configuración:
      storage:
        backend: encrypted_local
        remote: local_enc               # nombre del remote en .dvc/config
        encrypted_path: /mnt/secure     # ruta del volumen montado (para verificación)
        rev: v1.0                       # opcional
    """

    def __init__(self, cfg):
        self.remote: Optional[str] = getattr(cfg, "remote", None) or "local_enc"
        self.encrypted_path: Optional[str] = getattr(cfg, "encrypted_path", None)
        self.rev: Optional[str] = getattr(cfg, "rev", None)

    def resolve(self, path: str) -> Path:
        if self.encrypted_path:
            enc = Path(self.encrypted_path)
            if not enc.exists():
                raise RuntimeError(
                    f"EncryptedLocalBackend: la ruta encriptada '{self.encrypted_path}' "
                    "no existe o no está montada. "
                    "Desbloquea el volumen antes de iniciar el stack."
                )

        from .dvc_backend import DvcBackend

        class _Cfg:
            remote = self.remote
            rev = self.rev

        return DvcBackend(_Cfg()).resolve(path)
