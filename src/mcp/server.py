import sys
from pathlib import Path

# Add project root directory to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mcp.server import FastMCP

# Instantiate MCP Server
mcp = FastMCP("Financial-Python-REPL")


@mcp.tool()
def execute_python_calc(code: str) -> str:
    """
    Executes Python arithmetic code inside a restricted variable scope.
    Assign output to 'result' or define variables.
    """
    local_scope = {}
    safe_globals = {
        "__builtins__": {
            "abs": abs, "min": min, "max": max, "sum": sum,
            "len": len, "round": round, "float": float, "int": int
        }
    }

    try:
        exec(code, safe_globals, local_scope)
        res = local_scope.get("result", local_scope)
        return f"SUCCESS: {res}"
    except Exception as err:
        return f"EXECUTION_ERROR: {str(err)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")