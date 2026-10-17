"""redef-message: the redef items with the commit message and no code, as a control.

With redef (code only) and redef-commit (code and message) this separates what the code tells a model from what the
author's description does. It is not a realistic use: a reviewer always has the change itself.
"""

from .redef import (CATEGORIES, DEFAULT_N, DEFAULT_SEED, DISPLAY_CHARS, GROUPS, MODES, SOURCE,  # noqa: F401
                    build, instructions)

ID = "redef-message"
TITLE = "ReDef: commit only"
BLURB = ("Control: the same 200 ReDef changes with only the commit message (and file and function name), no code: how "
         "much can a model tell from the author's description alone?")


def state(item):
    return {"file": item["file_path"], "function": item["function_name"], "commit_message": item["commit_message"]}
