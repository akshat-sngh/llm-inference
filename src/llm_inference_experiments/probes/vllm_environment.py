"""Standalone optional-import probe run by the configured vLLM Python executable."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from typing import Any


def optional_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> dict[str, Any]:
    data: dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "vllm_version": optional_version("vllm"),
        "vllm_package_location": None,
        "torch_version": optional_version("torch"),
        "cuda_version": None,
        "cudnn_version": None,
        "cuda_available": None,
        "cuda_device_count": None,
        "selected_device_name": None,
        "selected_device_capability": None,
        "transformers_version": optional_version("transformers"),
        "tokenizers_version": optional_version("tokenizers"),
        "triton_version": optional_version("triton"),
        "numpy_version": optional_version("numpy"),
    }
    try:
        import vllm  # type: ignore[import-not-found]

        data["vllm_package_location"] = getattr(vllm, "__file__", None)
    except ImportError:
        pass
    try:
        import torch  # type: ignore[import-not-found]

        data["cuda_version"] = torch.version.cuda
        data["cudnn_version"] = torch.backends.cudnn.version()
        data["cuda_available"] = torch.cuda.is_available()
        data["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            data["selected_device_name"] = torch.cuda.get_device_name(0)
            data["selected_device_capability"] = list(torch.cuda.get_device_capability(0))
    except ImportError:
        pass
    return data


def packages() -> list[dict[str, str]]:
    distributions = [
        {"name": distribution.metadata["Name"], "version": distribution.version}
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    ]
    return sorted(distributions, key=lambda item: item["name"].lower().replace("-", "_"))


if __name__ == "__main__":
    command = sys.argv[1]
    print(json.dumps(environment() if command == "environment" else packages(), sort_keys=True))
