"""redef-commit: the redef items (same 200 changes, same labels) with the commit message sent alongside the diff.

redef is the on-the-fly case (an assistant edits a file; only the code change exists); this is the PR-review case,
where the author's description comes with the change. The difference between the two is what the message adds.
"""

from .redef import (CATEGORIES, DEFAULT_N, DEFAULT_SEED, DISPLAY_CHARS, GROUPS, MODES, SCORE_FLAG_AT,  # noqa: F401
                    SCORE_LABELS, SOURCE, build, instructions)

ID = "redef-commit"
TITLE = "ReDef: code + commit"
BLURB = ("The same 200 ReDef changes with the commit message (title and body) sent alongside the diff, as in a PR "
         "review: does it introduce a bug (noul), and what kind of change is it (choice)?")


def state(item):
    return {"file": item["file_path"], "function": item["function_name"], "commit_message": item["commit_message"],
            "diff": item["diff"]}
