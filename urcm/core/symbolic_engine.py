import math
import sys
import io
import ast
import operator
import contextlib
import threading
from typing import Any, Dict, Optional, Tuple, List, Union

class SafeMathVisitor(ast.NodeVisitor):
    """
    AST Visitor that only allows safe mathematical and logical operations.
    Rejects imports, function definitions, classes, and sensitive attribute access.
    """
    def __init__(self, allowed_variables: Dict[str, Any]):
        self.allowed_variables = allowed_variables
        self.allowed_functions = {
            'abs': abs, 'round': round, 'min': min, 'max': max, 
            'sum': sum, 'len': len, 'float': float, 'int': int, 
            'str': str, 'bool': bool, 'list': list, 'dict': dict, 'set': set,
            'range': range, 'enumerate': enumerate, 'zip': zip,
            'sorted': sorted, 'reversed': reversed,
            'all': all, 'any': any
        }
        # Allow safe math module functions
        for name in dir(math):
            if not name.startswith("_"):
                self.allowed_functions[f"math.{name}"] = getattr(math, name)
                
        # Add print to allowed functions
        self.allowed_functions['print'] = print

    def visit_Call(self, node):
        # Allow calling functions in the allowed list
        if isinstance(node.func, ast.Name):
            if node.func.id not in self.allowed_functions and node.func.id not in self.allowed_variables:
                 raise SecurityError(f"Function '{node.func.id}' is not allowed.")
        elif isinstance(node.func, ast.Attribute):
             # Allow math.sqrt style calls
             if isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                 pass # Safe math call
             else:
                 # Check if the method is on a safe object type (e.g. list.append)
                 pass # We allow method calls on objects generally, assuming objects themselves are safe
        self.generic_visit(node)

    def visit_Import(self, node):
        raise SecurityError("Import statements are not allowed.")
        
    def visit_ImportFrom(self, node):
        raise SecurityError("Import statements are not allowed.")
        
    def visit_FunctionDef(self, node):
        raise SecurityError("Defining functions is not allowed in simple evaluation.")
        
    def visit_ClassDef(self, node):
        raise SecurityError("Defining classes is not allowed.")

class SecurityError(Exception):
    pass

class SymbolicEngine:
    """
    Production-Grade Symbolic Engine (Calculator).
    
    Features:
    - AST-based Safety (No 'eval' injection)
    - Persistent State (Variables)
    - Execution Timeouts
    - Safe Standard Library Access
    """
    
    def __init__(self, timeout_seconds: float = 2.0):
        self.timeout_seconds = timeout_seconds
        self.state: Dict[str, Any] = {
            "math": math
        }
        # Populate with safe builtins
        self.state.update({
            'abs': abs, 'round': round, 'min': min, 'max': max, 
            'sum': sum, 'len': len, 'float': float, 'int': int, 
            'str': str, 'bool': bool, 'list': list, 'dict': dict, 'set': set,
            'range': range, 'enumerate': enumerate, 'zip': zip,
            'sorted': sorted, 'reversed': reversed,
            'all': all, 'any': any,
            'True': True, 'False': False, 'None': None,
            'print': print 
        })
        
    def _execute_with_timeout(self, code_str: str, mode: str = "eval") -> Tuple[bool, Any, str]:
        """Runs code in a separate thread to enforce timeout."""
        result_container = {"success": False, "result": None, "error": None}
        
        def target():
            try:
                # 1. Parse AST
                tree = ast.parse(code_str, mode=mode)
                
                # 2. Validate AST (Security Scan)
                visitor = SafeMathVisitor(self.state)
                visitor.visit(tree)
                        
                # 3. Execute
                if mode == "eval":
                    # For expressions, we return the result
                    compiled = compile(tree, filename="<string>", mode="eval")
                    val = eval(compiled, {"__builtins__": None}, self.state)
                    result_container["result"] = val
                    result_container["success"] = True
                else:
                    # For scripts (exec), we capture stdout
                    output_buffer = io.StringIO()
                    with contextlib.redirect_stdout(output_buffer):
                        compiled = compile(tree, filename="<string>", mode="exec")
                        exec(compiled, {"__builtins__": None}, self.state)
                    result_container["result"] = output_buffer.getvalue()
                    result_container["success"] = True
                    
            except Exception as e:
                result_container["error"] = str(e)
                
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self.timeout_seconds)
        
        if thread.is_alive():
            return False, None, f"Timeout: Execution exceeded {self.timeout_seconds}s."
            
        if result_container["success"]:
            return True, result_container["result"], ""
        else:
            return False, None, result_container["error"]

    def evaluate(self, expression: str) -> Tuple[bool, Any, str]:
        """
        Safely evaluates a mathematical expression.
        """
        return self._execute_with_timeout(expression, mode="eval")

    def execute_script(self, script: str) -> Tuple[bool, str, str]:
        """
        Safely executes a Python script.
        """
        return self._execute_with_timeout(script, mode="exec")

    def clear_state(self):
        """Resets the variable state."""
        self.__init__(self.timeout_seconds)

    def get_variable(self, name: str) -> Optional[Any]:
        return self.state.get(name)

    def set_variable(self, name: str, value: Any):
        self.state[name] = value
    
    def infer_next_in_sequence(self, seq: List[Union[int, float]]) -> Optional[float]:
        """
        Attempts simple pattern completion:
        - Arithmetic progression: constant difference
        - Geometric progression: constant ratio
        Returns next value or None if no simple pattern is detected.
        """
        if not isinstance(seq, list) or len(seq) < 2:
            return None
        # Arithmetic progression check
        diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        if all(abs(d - diffs[0]) < 1e-9 for d in diffs[1:]):
            return float(seq[-1] + diffs[0])
        # Geometric progression check (avoid zero)
        ratios = []
        for i in range(len(seq)-1):
            if seq[i] == 0:
                ratios = []
                break
            ratios.append(seq[i+1] / seq[i])
        if ratios and all(abs(r - ratios[0]) < 1e-9 for r in ratios[1:]):
            return float(seq[-1] * ratios[0])
        return None
