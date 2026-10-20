# Import libraries for parsing expressions and performing arithmetic
import ast
import operator

# Convert this function into a LangChain tool for the LLM
from langchain_core.tools import tool


# Allow only specific mathematical operators
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,       # Addition (+)
    ast.Sub: operator.sub,       # Subtraction (-)
    ast.Mult: operator.mul,      # Multiplication (*)
    ast.Div: operator.truediv,   # Division (/)
    ast.Pow: operator.pow,       # Power (**)
    ast.Mod: operator.mod,       # Modulo (%)
    ast.USub: operator.neg,      # Negative number (-5)
    ast.UAdd: operator.pos,      # Positive number (+5)
}


# Recursively calculate each part of the expression
def _calculate(node):

    # If the node is a number, return its value
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    # Handle operations involving two operands, such as 10 + 5
    if isinstance(node, ast.BinOp):

        # Find the corresponding allowed arithmetic function
        op = _ALLOWED_OPERATORS.get(type(node.op))

        if op is None:
            raise ValueError("Unsupported arithmetic operator.")

        # Calculate the left and right operands
        left = _calculate(node.left)
        right = _calculate(node.right)

        # Prevent excessively large exponents
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")

        # Perform the arithmetic operation
        result = op(left, right)

        # Reject complex numbers and excessively large results
        if isinstance(result, complex):
            raise ValueError("Complex results are not supported.")

        if isinstance(result, (int, float)) and abs(result) > 1e100:
            raise ValueError("Result is too large.")

        return result

    # Handle unary operations, such as -5 or +5
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPERATORS.get(type(node.op))

        if op is None:
            raise ValueError("Unsupported unary operator.")

        return op(_calculate(node.operand))

    # Reject anything other than supported arithmetic
    raise ValueError("Only basic arithmetic is supported.")


# Register the calculator as a tool the LLM can call
@tool
def calculator(expression: str) -> str:
    """Calculate arithmetic expressions accurately.
    Supports addition, subtraction, multiplication, division,
    modulo, and powers.
    """

    try:
        # Parse the expression into an AST without executing Python code
        tree = ast.parse(expression, mode="eval")

        # Calculate the expression and return the result as text
        result = _calculate(tree.body)
        return str(result)

    # Handle invalid calculations gracefully
    except ZeroDivisionError:
        return "Error: division by zero."

    except (ValueError, SyntaxError, TypeError, OverflowError) as error:
        return f"Calculation error: {error}"