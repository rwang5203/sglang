# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Diffusion pipelines for sglang.multimodal_gen.

This package contains diffusion pipelines for generating videos and images.
"""

from typing import TYPE_CHECKING, Any, cast

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import Backend, ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
        ComposedPipelineBase,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
        LoRAPipeline,
    )
    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        maybe_download_model as maybe_download_model,
    )
    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        verify_model_config_and_directory as verify_model_config_and_directory,
    )

    class PipelineWithLoRA(LoRAPipeline, ComposedPipelineBase):
        pass


logger = init_logger(__name__)


# The pipeline classes import the model runtime, which a process that only
# needs Req, such as a request client, does not load.
def __getattr__(name: str) -> Any:
    if name == "ComposedPipelineBase":
        from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
            ComposedPipelineBase,
        )

        value = ComposedPipelineBase
    elif name == "LoRAPipeline":
        from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
            LoRAPipeline,
        )

        value = LoRAPipeline
    elif name == "PipelineWithLoRA":
        value = type(
            name,
            (__getattr__("LoRAPipeline"), __getattr__("ComposedPipelineBase")),
            {
                "__module__": __name__,
                "__doc__": "Type for a pipeline that has both ComposedPipelineBase "
                "and LoRAPipeline functionality.",
            },
        )
    elif name in ("maybe_download_model", "verify_model_config_and_directory"):
        from sglang.multimodal_gen.runtime.utils import hf_diffusers_utils

        value = getattr(hf_diffusers_utils, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def build_pipeline(
    server_args: ServerArgs,
) -> "PipelineWithLoRA":
    """
    Only works with valid hf diffusers configs. (model_index.json)
    We want to build a pipeline based on the inference args mode_path:
    1. download the model from the hub if it's not already downloaded
    2. verify the model config and directory
    3. based on the config, determine the pipeline class
    """
    from sglang.multimodal_gen.registry import discover_pipelines, get_model_info

    model_path = server_args.model_path

    # Check if pipeline class is explicitly specified
    if server_args.pipeline_class_name:
        from sglang.multimodal_gen.registry import (
            get_pipeline_class,
            get_registered_pipeline_names,
        )

        available_pipelines = get_registered_pipeline_names()
        logger.info(f"Requested pipeline_class_name: {server_args.pipeline_class_name}")
        logger.info(f"Available pipelines in registry: {available_pipelines}")
        pipeline_cls = get_pipeline_class(server_args.pipeline_class_name)
        if pipeline_cls is None:
            raise ValueError(
                f"Pipeline class '{server_args.pipeline_class_name}' not found in registry. "
                f"Available pipelines: {available_pipelines}"
            )
        logger.info(
            f"✓ Using explicitly specified pipeline: {server_args.pipeline_class_name} (class: {pipeline_cls.__name__})"
        )
    else:
        logger.info("No pipeline_class_name specified, using model_index.json")

        # The pipeline is built right away, so discover the pipelines first.
        # Otherwise resolution with the sglang backend would defer their
        # import and read every pipeline source to name the classes.
        if server_args.backend == Backend.SGLANG:
            discover_pipelines()
        model_info = get_model_info(
            model_path,
            backend=server_args.backend,
            model_id=server_args.model_id,
        )
        pipeline_cls = model_info.pipeline_cls
        logger.info(f"Using pipeline from model_index.json: {pipeline_cls.__name__}")

    # instantiate the pipelines
    pipeline = pipeline_cls(model_path, server_args)

    logger.info("Pipeline instantiated")

    return cast("PipelineWithLoRA", pipeline)


__all__ = [
    "build_pipeline",
    "ComposedPipelineBase",
    "Req",
    "LoRAPipeline",
]
