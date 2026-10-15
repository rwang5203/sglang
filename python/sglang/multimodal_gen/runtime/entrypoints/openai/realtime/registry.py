# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
        BaseRealtimeModelAdapter,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


# A built-in adapter is registered by its import path, as an adapter module
# imports the pipeline runtime. Only a config served in realtime imports one.
_REALTIME_ADAPTER_REGISTRY: dict[type, type[BaseRealtimeModelAdapter] | str] = {}
_BUILTIN_ADAPTERS_REGISTERED = False
_BUILTIN_ADAPTERS = "sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters"


def register_realtime_model_adapter(
    pipeline_config_cls: type,
    adapter_cls: type[BaseRealtimeModelAdapter],
) -> None:
    _REALTIME_ADAPTER_REGISTRY[pipeline_config_cls] = adapter_cls


def _register_builtin_realtime_model_adapters() -> None:
    global _BUILTIN_ADAPTERS_REGISTERED
    if _BUILTIN_ADAPTERS_REGISTERED:
        return

    from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
        LingBotWorldCausalDMDConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import (
        SanaWMRealtimeConfig,
    )

    _REALTIME_ADAPTER_REGISTRY[LingBotWorldCausalDMDConfig] = (
        f"{_BUILTIN_ADAPTERS}.lingbot_world_realtime_adapter:LingBotWorldRealtimeAdapter"
    )
    _REALTIME_ADAPTER_REGISTRY[SanaWMRealtimeConfig] = (
        f"{_BUILTIN_ADAPTERS}.sana_wm_realtime_adapter:SanaWMRealtimeAdapter"
    )
    _BUILTIN_ADAPTERS_REGISTERED = True


def _registered_adapter_cls(
    pipeline_config_cls: type,
) -> type[BaseRealtimeModelAdapter] | None:
    adapter_cls = _REALTIME_ADAPTER_REGISTRY.get(pipeline_config_cls)
    if isinstance(adapter_cls, str):
        adapter_cls = pkgutil.resolve_name(adapter_cls)
        _REALTIME_ADAPTER_REGISTRY[pipeline_config_cls] = adapter_cls
    return adapter_cls


def get_realtime_model_adapter(
    server_args: ServerArgs,
) -> BaseRealtimeModelAdapter:
    _register_builtin_realtime_model_adapters()

    pipeline_config = server_args.pipeline_config
    for config_cls in type(pipeline_config).__mro__:
        adapter_cls = _registered_adapter_cls(config_cls)
        if adapter_cls is not None:
            return adapter_cls()

    raise ValueError(
        "Realtime video is not supported for pipeline config "
        f"{type(pipeline_config).__name__}; no realtime adapter is registered."
    )


def has_realtime_model_adapter(server_args: ServerArgs) -> bool:
    """Whether the resolved pipeline config has a registered adapter."""
    _register_builtin_realtime_model_adapters()
    for config_cls in type(server_args.pipeline_config).__mro__:
        if config_cls in _REALTIME_ADAPTER_REGISTRY:
            # Startup checks call this, so a realtime server loads its adapter
            # here rather than in its first session.
            _registered_adapter_cls(config_cls)
            return True
    return False
