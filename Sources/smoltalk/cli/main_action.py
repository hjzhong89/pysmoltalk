from enum import StrEnum

import inquirer
from logger import get_logger

from smoltalk.cli.export_action import handle_export_action

logger = get_logger()


class MainActionQuestion(StrEnum):
    """
    Options available from the Main app
    """
    EXPORT = "Export LLM Model"

    @staticmethod
    def from_label(label: str):
        for action in MainActionQuestion:
            if label == action:
                return action

        raise NotImplementedError()


def prompt_for_action():
    """
    First question for the user
    :return:
    """
    action_key = "action"

    questions = [
        inquirer.List(action_key,
                      message="Main Menu",
                      choices=[choice for choice in MainActionQuestion]
                      )
    ]

    answers = inquirer.prompt(questions)
    chosen = MainActionQuestion.from_label(answers[action_key])

    match chosen:
        case MainActionQuestion.EXPORT:
            handle_export_action()
        case _:
            logger.info("Other")
