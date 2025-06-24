from __future__ import annotations

import ast
import itertools as it
from pathlib import Path

import black
import isort


def top_level_fns(module: ast.Module) -> list[str]:
    return [
        f.name
        for f in module.body
        if isinstance(f, ast.FunctionDef) and not f.name.startswith("_")
    ]


if __name__ == "__main__":
    components = Path(__file__).parent
    files = list(components.glob("[!_.]*"))

    fns_by_files = {}
    for file in files:
        with file.open() as fp:
            body = ast.parse(fp.read())
        fns_by_files[file.name.removesuffix(".py")] = top_level_fns(body)

    with (components / "__init__.py").open("w+") as fp:
        fp.write(
            black.format_str(
                isort.api.sort_code_string(
                    "\n".join(
                        f"from .{name} import ({','.join(fns)})"
                        for name, fns in fns_by_files.items()
                    )
                ),
                mode=black.Mode(),  # type: ignore
            )
        )

        fp.write(
            black.format_str(
                f"__all__ = {list(it.chain.from_iterable(fns_by_files.values()))}",
                mode=black.Mode(),  # type: ignore
            )
        )
