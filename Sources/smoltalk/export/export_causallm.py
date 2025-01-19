import datetime
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from logger import get_logger
from transformers import LlamaConfig

from smoltalk.models.cached_causallm import KvCacheStateLlamaForCausalLM

logger = get_logger()
system_prompt_rewrite = "You are an AI writing assistant. Your task is to rewrite the user's email to make it more professional and approachable while maintaining its main points and key message. Do not return any text other than the rewritten message."
user_prompt_rewrite = "Rewrite the message below to make it more friendly and approachable while maintaining its main points and key message. Do not add any new information or return any text other than the rewritten message\nThe message:"
messages = [{"role": "system", "content": system_prompt_rewrite},
            {"role": "user",
             "content": f"{user_prompt_rewrite} The CI is failing after your last commit!"}]


def export_causallm(pretrained_dir_or_name: str,
                    export_dir: Path,
                    batch_size: int = 1,
                    context_size: int = 4096,
                    device: str = "mps"):
    """
    Convert a Llama-type, Causal PyTorch LLM to MLModel format with CoreML Tools
    :param pretrained_dir_or_name: Passed to the underlying HF model's #from_pretrained method
    :param export_dir: The parent dir to export models to
    :param batch_size: The max batch size for model inferencing
    :param context_size: The max context size used by the cache
    :param device: "mps" or "cpu"
    :return:
    """
    logger.info(f"Loading KvCacheStateLlamaForCausalLM: {pretrained_dir_or_name}")
    now = datetime.date.today().strftime("%m%d%Y")

    torch_model = KvCacheStateLlamaForCausalLM(pretrained_dir_or_name,
                                               context_size=context_size,
                                               device=device)

    # Initialize and trace PyTorch model
    example_inputs: tuple[torch.Tensor, ...] = (
        torch.zeros((1, 2), dtype=torch.int32).to(device),
        torch.zeros((1, 1, 2, 5), dtype=torch.float32).to(device)
    )
    traced_model: torch.jit.ScriptModule = torch.jit.trace(
        torch_model.to(device).eval(), example_inputs=example_inputs
    )

    # Convert to Core ML
    query_size = ct.RangeDim(lower_bound=1, upper_bound=context_size, default=1)
    final_step = ct.RangeDim(lower_bound=1, upper_bound=context_size, default=1)
    inputs: list[ct.TensorType] = [
        ct.TensorType(shape=(batch_size, query_size), dtype=np.int32, name="inputIds"),
        ct.TensorType(
            shape=(batch_size, 1, query_size, final_step),
            dtype=np.float16,
            name="causalMask",
        ),
    ]
    states: list[ct.StateType] = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
            name="valueCache",
        ),
    ]

    outputs: list[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
    mlmodel: ct.models.MLModel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=True,
        compute_units=ct.ComputeUnit.ALL,
    )
    loaded_config: LlamaConfig = torch_model.config
    terminating_tokens = loaded_config.eos_token_id
    mlmodel.author = "Jaime Zhong"
    mlmodel.short_description = f"{KvCacheStateLlamaForCausalLM}"
    mlmodel.version = f"v0.0.0-{now}"

    mlmodel.input_description["inputIds"] = "Flexible shaped tokenized inputs"
    mlmodel.input_description["causalMask"] = f"[1, 1, [1, {context_size}], [1, {context_size}]] causal mask"
    mlmodel.output_description["logits"] = f"[batchSize, contextSize, vocabSize]; terminating tokens: {terminating_tokens}"

    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=128,
    )
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel_optimized = ct.optimize.coreml.linear_quantize_weights(
        mlmodel, config=config
    )

    out_path = export_dir.joinpath("PySmolTalk/dist/PySmolTalkLM.mlpackage")
    mlmodel_optimized.save(str(out_path))
