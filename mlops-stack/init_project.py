#!/usr/bin/env python3
"""
init_project.py
───────────────
Script de inicialización para un nuevo caso de uso en el stack MLOps.
Lee la configuración (MODEL_CONFIG o --config) y prepara el backend de
almacenamiento: inicializa DVC si es necesario, configura el remote, y
trackea el archivo de datos con `dvc add`.

Uso:
  python init_project.py                            # usa MODEL_CONFIG o iris por defecto
  python init_project.py --config configs/mi.yaml  # caso de uso específico
  python init_project.py --non-interactive          # sin preguntas (CI/CD)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=check)


def _dvc_init():
    if Path(".dvc").exists():
        print("[DVC] .dvc/ ya existe — omitiendo dvc init.")
        return
    print("[DVC] Inicializando repositorio DVC...")
    _run(["dvc", "init"])
    _run(["git", "add", ".dvc", ".dvcignore"])
    print("[DVC] Repo DVC creado. Recuerda hacer 'git commit' de los archivos generados.")


def _ask(prompt: str, default: str = "") -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def _configure_remote(remote_name: str, backend: str, interactive: bool):
    print(f"\n[DVC] Configuración del remote '{remote_name}' ({backend}):")
    if backend == "dvc":
        if interactive:
            url = _ask("  URL del remote DVC (ej: s3://bucket/ruta  o  ssh://host:/ruta)")
            if url:
                _run(["dvc", "remote", "add", "-d", remote_name, url])
        else:
            print("  Configura manualmente:")
            print(f"    dvc remote add -d {remote_name} s3://mi-bucket/dvc-cache")
            print(f"    dvc remote add -d {remote_name} ssh://usuario@host:/ruta/dvc-cache")

    elif backend == "azure":
        if interactive:
            url = _ask("  URL del contenedor Azure (ej: azure://mi-contenedor/prefijo)")
            if url:
                _run(["dvc", "remote", "add", "-d", remote_name, url])
                _run(["dvc", "remote", "modify", remote_name,
                      "connection_string", "${AZURE_STORAGE_CONNECTION_STRING}"])
        print("  Exporta en producción:")
        print("    export AZURE_STORAGE_CONNECTION_STRING='DefaultEndpoints...'")
        print("  O usa Managed Identity (AZURE_CLIENT_ID + AZURE_CLIENT_SECRET + AZURE_TENANT_ID).")

    elif backend == "gdrive":
        if interactive:
            folder_id = _ask("  ID de la carpeta Google Drive (de la URL de Drive)")
            if folder_id:
                _run(["dvc", "remote", "add", "-d", remote_name, f"gdrive://{folder_id}"])
        print("  Para CI/Docker sin interacción usa Service Account:")
        print(f"    dvc remote modify {remote_name} gdrive_use_service_account true")
        print(f"    dvc remote modify {remote_name} gdrive_service_account_json_file_path "
              "/run/secrets/gdrive-sa.json")
        print("  Monta el archivo SA como secreto Docker o variable GDRIVE_CREDENTIALS_DATA.")

    elif backend == "encrypted_local":
        if interactive:
            enc_path = _ask("  Ruta del volumen encriptado montado (ej: /mnt/secure/dvc-cache)")
            if enc_path:
                _run(["dvc", "remote", "add", "-d", remote_name, enc_path])
        print("  Asegúrate de montar el volumen ANTES de docker compose up.")
        print("  En docker-compose.yml añade:")
        print("    volumes:")
        print("      - /mnt/secure:/mnt/secure:ro")

    # Confirmar con git
    dvc_config = Path(".dvc/config")
    if dvc_config.exists():
        _run(["git", "add", ".dvc/config"], check=False)


def _dvc_add_data(data_path: str):
    p = Path(data_path)
    if not p.exists():
        print(f"\n[DVC] '{data_path}' no encontrado en disco.")
        print("  Opciones:")
        print(f"    a) Coloca el archivo en '{data_path}' y vuelve a ejecutar.")
        print(f"    b) Ejecuta 'dvc pull' si el archivo ya está en el remote.")
        return

    print(f"\n[DVC] Añadiendo '{data_path}' al tracking de DVC...")
    p.parent.mkdir(parents=True, exist_ok=True)
    _run(["dvc", "add", data_path])
    dvc_file = data_path + ".dvc"
    gitignore = str(p.parent / ".gitignore")
    _run(["git", "add", dvc_file, gitignore], check=False)
    print(f"[DVC] '{dvc_file}' generado y staged.")


def main():
    parser = argparse.ArgumentParser(
        description="Inicializa el backend de almacenamiento para un caso de uso MLOps."
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("MODEL_CONFIG", "configs/iris-classifier.yaml"),
        help="Ruta al archivo YAML del caso de uso.",
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Omite las preguntas interactivas (útil en CI/CD).",
    )
    args = parser.parse_args()
    interactive = not args.non_interactive

    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml no instalado. Ejecuta: pip install pyyaml")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config '{config_path}' no encontrada.")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ds = cfg.get("data_source", {})
    storage = ds.get("storage", {})
    backend = storage.get("backend", "git_local")
    data_path = ds.get("path", "")
    remote_name = storage.get("remote", "myremote")

    print(f"\n{'='*55}")
    print(f"  MLOps Stack — Inicialización de proyecto")
    print(f"{'='*55}")
    print(f"  Config  : {config_path}")
    print(f"  Backend : {backend}")
    print(f"  Dataset : {data_path or '(sklearn dataset — sin archivo)'}")
    print(f"{'='*55}\n")

    if backend == "git_local":
        print("[git_local] Sin configuración adicional requerida.")
        print("  El archivo de datos debe estar git-tracked o montado como volumen Docker.")
        print(f"  Ruta esperada: {data_path}")
        return

    # Todos los backends distintos de git_local requieren DVC
    _dvc_init()
    _configure_remote(remote_name, backend, interactive)

    if data_path:
        _dvc_add_data(data_path)

    print(f"\n{'='*55}")
    print("  Pasos siguientes")
    print(f"{'='*55}")
    print("  1. git commit -m 'chore: configure DVC storage backend'")
    print("  2. git push")
    print(f"  3. dvc push --remote {remote_name}   # sube los datos al remote")
    print()
    print("  En producción (antes de docker compose up):")
    if backend == "azure":
        print("  - export AZURE_STORAGE_CONNECTION_STRING='...'")
    elif backend == "gdrive":
        print("  - Montar el archivo de Service Account:")
        print("      docker secret create gdrive-sa /ruta/local/sa.json")
    elif backend == "encrypted_local":
        print("  - Montar y desbloquear el volumen encriptado antes de docker compose up.")
    print()
    print("  4. docker compose up -d")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
