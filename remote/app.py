import subprocess
from pathlib import Path, PurePosixPath
from typing import Iterable

from grpclib import GRPCError, Status
import modal


APP_NAME = "nv_app"
VOLUME_NAME = "nv_vol"
BASE_IMAGE = "pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel"
IMAGE_ENV = {
    "HF_HOME": "/workspace/hf",
    "HUGGINGFACE_HUB_CACHE": "/workspace/hf",
}
RUN_COMMANDS = [
    "apt-get update && apt-get install -y curl ca-certificates gnupg",
    "curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
    "dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb",
    "apt-get update",
    "curl -fsSL -o /tmp/nsight-compute_2025.3.0_amd64.deb https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nsight-compute-2025.3.0_2025.3.0.19-1_amd64.deb",
    "dpkg -i /tmp/nsight-compute_2025.3.0_amd64.deb || apt-get -f install -y",
    "rm -f /tmp/nsight-compute_2025.3.0_amd64.deb",
    "apt-get update",
    "apt-get install -y nsight-systems-2025.3.2",
]
APT_PACKAGES = [
    "graphviz",
]
UV_PACKAGES = [
    "nvtx",
    "vllm",
    "pydot",
    "accelerate",
    "nvidia-cutlass",
    "nvidia-cutlass-dsl",
]
GPU_DEFAULT = "B200"
GPU_ALIASES = {
    "L4": "L4",
    "L40S": "L40S",
    "A100": "A100",
    "H100": "H100",
    "B200": "B200"
}

LOCAL_WORKSPACE = Path("nvfp4")
LOCAL_OUTPUTS = Path("out_local")
VOLUME_MOUNT_PATH = PurePosixPath("/workspace")


def _to_volume_path(container_path: PurePosixPath) -> PurePosixPath:
    try:
        relative = container_path.relative_to(VOLUME_MOUNT_PATH)
    except ValueError:
        return container_path

    if str(relative) in {"", "."}:
        return PurePosixPath("/")

    return PurePosixPath("/") / relative


REMOTE_WORKSPACE = PurePosixPath("/workspace/nvfp4")
REMOTE_OUTPUTS = PurePosixPath("/workspace/out_local")
VOLUME_WORKSPACE_PATH = _to_volume_path(REMOTE_WORKSPACE)
VOLUME_OUTPUTS_PATH = _to_volume_path(REMOTE_OUTPUTS)


def _build_image() -> modal.Image:
    image = modal.Image.from_registry(BASE_IMAGE).env(IMAGE_ENV)

    for command in RUN_COMMANDS:
        image = image.run_commands(command)

    if APT_PACKAGES:
        image = image.apt_install(*APT_PACKAGES)

    image = image.run_commands("pip install --upgrade pip uv")
    if UV_PACKAGES:
        image = image.run_commands("uv pip install --system " + " ".join(UV_PACKAGES))

    return image


def _gpu_type(name: str | None = None) -> str:
    alias = name or GPU_DEFAULT
    return GPU_ALIASES.get(alias, alias)


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = _build_image()


@app.function(image=image, volumes={"/workspace": volume}, gpu=_gpu_type())
def build_image_remote() -> None:
    """Build the container image on Modal."""


def _download_entries(entries: Iterable[modal.volume.FileEntry]) -> int:
    downloaded = 0
    for entry in entries:
        remote_path = PurePosixPath(entry.path)
        try:
            rel_path = remote_path.relative_to(VOLUME_OUTPUTS_PATH)
        except ValueError:
            rel_path = remote_path

        local_target = LOCAL_OUTPUTS / Path(rel_path.as_posix())

        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            local_target.mkdir(parents=True, exist_ok=True)
            continue

        if entry.type != modal.volume.FileEntryType.FILE:
            continue

        local_target.parent.mkdir(parents=True, exist_ok=True)
        with local_target.open("wb") as fh:
            for chunk in volume.read_file(entry.path):
                fh.write(chunk)
        downloaded += 1

    return downloaded


def _sync_outputs_impl(verbose: bool = True) -> int:
    try:
        entries = volume.listdir(VOLUME_OUTPUTS_PATH.as_posix(), recursive=True)
    except (FileNotFoundError, GRPCError) as exc:
        if isinstance(exc, GRPCError) and exc.status is not Status.NOT_FOUND:
            raise
        if verbose:
            print(f"No outputs at {REMOTE_OUTPUTS}; nothing to download.")
        return 0

    if not entries:
        if verbose:
            print(f"No outputs at {REMOTE_OUTPUTS}; nothing to download.")
        return 0

    LOCAL_OUTPUTS.mkdir(parents=True, exist_ok=True)
    downloaded = _download_entries(entries)
    if verbose:
        print(f"Downloaded {downloaded} file(s) from {REMOTE_OUTPUTS} into {LOCAL_OUTPUTS}")
    return downloaded


@app.local_entrypoint()
def sync_workspace() -> None:
    if not LOCAL_WORKSPACE.is_dir():
        raise FileNotFoundError(f"Missing local workspace directory: {LOCAL_WORKSPACE}")

    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(LOCAL_WORKSPACE), VOLUME_WORKSPACE_PATH.as_posix())

    print(f"Uploaded {LOCAL_WORKSPACE} -> {REMOTE_WORKSPACE} (volume {volume.name})")


@app.local_entrypoint()
def sync_outputs() -> None:
    _sync_outputs_impl(verbose=True)


@app.function(image=image, volumes={"/workspace": volume}, gpu=_gpu_type())
def run_profile_script() -> None:
    subprocess.run(["/bin/bash", "profile_test.sh"], cwd="/workspace/nvfp4", check=True)


@app.local_entrypoint()
def profile_and_fetch() -> None:
    run_profile_script.remote()
    downloaded = _sync_outputs_impl(verbose=False)
    if downloaded:
        print(f"Profile artifacts pulled into {LOCAL_OUTPUTS}")
    else:
        print("Profile completed, but no outputs were found to download.")


@app.function(image=image, volumes={"/workspace": volume})
def volume_shell() -> None:
    """Dummy function spec for opening a Modal shell with the workspace volume mounted."""


@app.function(image=image, volumes={"/workspace": volume}, gpu=_gpu_type())
def gpu_shell() -> None:
    """Dummy function spec for opening a Modal shell with both the volume and GPU attached."""


__all__ = [
    "app",
    "image",
    "volume",
    "build_image_remote",
    "run_profile_script",
    "profile_and_fetch",
    "volume_shell",
    "volume_gpu_shell",
    "sync_workspace",
    "sync_outputs",
]
