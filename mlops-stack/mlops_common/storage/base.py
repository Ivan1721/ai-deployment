"""
base.py
───────
Contrato de backend de almacenamiento: cada implementación concreta sabe cómo
materializar un archivo de datos en disco local, sin importar dónde reside
originalmente (git, DVC+S3, DVC+Azure, Google Drive, carpeta encriptada, etc.).

El patrón usado es Strategy+Adapter:
  - Strategy: la interfaz StorageBackend es intercambiable según la config.
  - Adapter: cada subclase adapta una API externa (DVC, Azure SDK, etc.)
    al contrato uniform resolve(path) -> Path.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Garantiza que un archivo de datos esté disponible localmente.

    resolve(path) -> Path:
        Recibe la ruta lógica del archivo (como aparece en el YAML de config)
        y devuelve la ruta local donde el archivo puede leerse.
        Si el backend necesita descargar/desencriptar el archivo, lo hace aquí.
    """

    @abstractmethod
    def resolve(self, path: str) -> Path:
        ...
