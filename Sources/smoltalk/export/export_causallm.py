import datetime
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from logger import get_logger
from transformers import LlamaConfig, PreTrainedTokenizer, AutoTokenizer

from smoltalk.models.cached_causallm import KvCacheStateLlamaForCausalLM

logger = get_logger()
now = datetime.date.today().strftime("%m%d%Y")


def load_model(pretrained_dir_or_name: str,
               context_size: int = 4096,
               device: str = "mps") -> (KvCacheStateLlamaForCausalLM, PreTrainedTokenizer):
    logger.info(f"Loading KvCacheStateLlamaForCausalLM: {pretrained_dir_or_name}")

    torch_model = KvCacheStateLlamaForCausalLM(pretrained_dir_or_name,
                                               context_size=context_size,
                                               device=device)
    torch_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir_or_name)


def create_inputs(tokenizer: PreTrainedTokenizer | None = None,
                  device: str = "mps",
                  zeros: bool = False) -> (torch.Tensor, torch.Tensor):
    if zeros:
        # Initialize and trace PyTorch model
        example_inputs = torch.zeros((1, 2), dtype=torch.int32).to(device)
        causal_mask = torch.zeros((1, 1, 2, 5), dtype=torch.float32).to(device)
        return example_inputs, causal_mask
    else:
        input_text = "Please write a fun, casual, and appetizing description for miso soup.<think>\n"
        inputs: torch.Tensor = tokenizer.encode(input_text, return_tensors="pt")
        token_count = inputs.size(dim=1)
        logger.info(f"Input Shape: {inputs.shape}")

        logger.info("Creating causal mask")
        causal_mask = [[[]]]
        for i in range(token_count):
            layer = [0 if j <= i else -torch.inf for j in range(token_count)]
            causal_mask[0][0].append(layer)

        print(f"Created causal mask: {causal_mask}")
        causal_mask = torch.tensor(causal_mask)
        logger.info(f"Created causal Mask: {causal_mask.shape}")

        return inputs, causal_mask


def trace_model(torch_model: torch.nn.Module,
                example_inputs: torch.Tensor,
                causal_mask: torch.Tensor,
                export_dir: Path,
                device: str = "mps") -> torch.jit.ScriptModule:

    traced_model: torch.jit.ScriptModule = torch.jit.script(torch_model.to(device).eval(),
                                                            example_inputs=[(example_inputs, causal_mask)])

    traced_model_path = export_dir.joinpath(f"pysmoltalk_traced_{now}.pt")
    torch.save(traced_model, traced_model_path)
    return traced_model


def export_causallm(pretrained_dir_or_name: str,
                    export_dir: Path,
                    batch_size: int = 1,
                    context_size: int = 4096,
                    device: str = "mps"):
    """
    Convert a Causal PyTorch LLM to MLModel format with CoreML Tools
    :param pretrained_dir_or_name: Passed to the underlying HF model's #from_pretrained method
    :param export_dir: The parent dir to export models to
    :param batch_size: The max batch size for model inferencing
    :param context_size: The max context size used by the cache
    :param device: "mps" or "cpu"
    :return:
    """
    (torch_model, tokenizer) = load_model(pretrained_dir_or_name, context_size, device)

    (example_inputs, causal_mask) = create_inputs(tokenizer)

    traced_model = trace_model(torch_model,
                               example_inputs,
                               causal_mask,
                               export_dir,
                               device)


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
    mlmodel.output_description[
        "logits"] = f"[batchSize, contextSize, vocabSize]; terminating tokens: {terminating_tokens}"

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
