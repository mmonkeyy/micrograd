"""Run notebook code cells in order without a Jupyter server."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat

ROOT = Path(__file__).resolve().parent

def run_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = "\n".join(line for line in cell.source.splitlines() if not line.startswith("%"))
        code = ast.parse(source)
        last = code.body.pop() if code.body and isinstance(code.body[-1], ast.Expr) else None
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(code, str(path), "exec"), namespace)
            result = eval(compile(ast.Expression(last.value), str(path), "eval"), namespace) if last else None
        count += 1
        cell.execution_count = count
        cell.outputs = []
        if stream.getvalue():
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=stream.getvalue()))
        if result is not None:
            cell.outputs.append(nbformat.v4.new_output("execute_result", execution_count=count, data={"text/plain": repr(result)}))
        plt.close("all")
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    if path.name == "backprop.ipynb":
        assert namespace["loss"].data < 0.05, "Training loss did not decrease sufficiently"
        print(f"{path.name}: {count} cells passed, loss={namespace['loss'].data:.6f}")
    else:
        print(f"{path.name}: {count} cells passed")

if __name__ == "__main__":
    os.chdir(ROOT)
    for name in ("backprop.ipynb", "makemore_bigram.ipynb", "makemore_mlp.ipynb"):
        run_notebook(ROOT / name)
