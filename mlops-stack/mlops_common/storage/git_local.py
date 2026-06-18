"""
git_local.py
────────────
Backend por defecto: el archivo ya está en disco, trackeado por git o montado
como volumen Docker. No realiza ninguna descarga; solo verifica que exista.
"""

from pathlib import Path

from .base import StorageBackend


class GitLocalBackend(StorageBackend):
    """No-op: devuelve la ruta tal cual, comprobando que el archivo existe."""

    def resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"GitLocalBackend: archivo no encontrado en '{path}'. "
                "Asegúrate de que esté git-tracked y montado en el contenedor "
                "(volumes: en docker-compose.yml)."
            )
        return p
