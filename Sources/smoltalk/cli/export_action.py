from enum import StrEnum
from pathlib import Path

import inquirer
from logger import get_logger

from smoltalk.export import export_causallm

logger = get_logger()


def get_model_source() -> str:
    """
    Prompt the user for the model to convert. Users are allowed to provide a local model,
    a known model from HF, or an alternate model from HF
    :return:
    """

    class ModelSourceQuestion(StrEnum):
        LOCAL = "Local Pretrained Model",
        SMOLLM = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        LLAMA = "meta-llama/Llama-3.1-8B-Instruct",
        OTHER = "Other Repository Model"

        @staticmethod
        def from_label(label: str):
            for _source in ModelSourceQuestion:
                if label == _source:
                    return _source

            raise NotImplementedError()

    response: dict[str, str] = inquirer.prompt([
        inquirer.List("source",
                      message="Select an Instruction LLM to export",
                      choices=[s for s in ModelSourceQuestion])
    ])

    match (source := ModelSourceQuestion.from_label(response["source"])):
        case ModelSourceQuestion.LOCAL:
            response = inquirer.prompt([
                inquirer.Path('source_dir',
                              message="Provide existing pretrained model directory",
                              path_type=inquirer.Path.DIRECTORY,
                              exists=True)
            ])
            return response["source_dir"]
        case ModelSourceQuestion.OTHER:
            response = inquirer.prompt([
                inquirer.Text('pretrained_model',
                              message="Provide Model Repository Name")
            ])
            return response["source_dir"]
        case _:
            return source


def get_export_dir() -> str:
    """

    :return:
    """
    response = inquirer.prompt([
        inquirer.Path('export_dir',
                      message="Provide existing parent directory for exported model. Model will be saved to a new subdirectory.",
                      path_type=inquirer.Path.DIRECTORY,
                      exists=True)
    ])

    return response["export_dir"]


def handle_export_action():
    """
    Export a Pytorch Causal LLM model to CoreML format.

    Get Export Parameters from user:
    1. Get the model source (i.e. local dir, HF pretrained model name)
    2. Get the converted model save location; if not provided then use default MODELS_DIR
    3. Get the conversion parameters (batch size, max context size, optimization parameters)

    Do conversion
    :return:
    """
    model_source = get_model_source()
    export_dir = Path(get_export_dir())
    export_causallm.export_causallm(model_source, export_dir)
