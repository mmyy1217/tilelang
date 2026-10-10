"""Extract a standalone, specialized kernel from a built-in case factory."""

import ast
import inspect
from functools import partial
from pathlib import Path


def resolve_factory(factory):
    args, kwargs = (), {}
    if isinstance(factory, partial):
        args, kwargs = factory.args, factory.keywords
        factory = factory.func
    if factory.__name__ == "<lambda>":
        tree = ast.parse(Path(inspect.getsourcefile(factory)).read_text())
        expression = next(
            node for node in ast.walk(tree) if isinstance(node, ast.Lambda) and node.lineno == factory.__code__.co_firstlineno
        ).body
        if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
            raise ValueError("Kernel snapshot requires a lambda calling a named factory")
        args = tuple(ast.literal_eval(arg) for arg in expression.args)
        kwargs = {item.arg: ast.literal_eval(item.value) for item in expression.keywords}
        factory = factory.__globals__[expression.func.id]
    bound = inspect.signature(factory).bind(*args, **kwargs)
    bound.apply_defaults()
    return factory, dict(bound.arguments)


def constant_value(node):
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.Load,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.IfExp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.FloorDiv,
        ast.Mod,
        ast.Not,
        ast.USub,
        ast.UAdd,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
    )
    if not all(isinstance(item, allowed) for item in ast.walk(node)):
        raise ValueError("Expression is not a supported compile-time constant")
    return eval(compile(ast.fix_missing_locations(ast.Expression(node)), "<snapshot>", "eval"), {"__builtins__": {}})


def literal(value):
    return ast.parse(repr(value), mode="eval").body


class Specialize(ast.NodeTransformer):
    def __init__(self, bindings):
        self.bindings = bindings

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.bindings:
            return ast.copy_location(literal(self.bindings[node.id]), node)
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        try:
            condition = constant_value(node.test)
        except ValueError:
            return node
        return node.body if condition else node.orelse

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        try:
            condition = constant_value(node.test)
        except ValueError:
            return node
        return node.body if condition else node.orelse


def standalone_kernel(factory):
    factory, bindings = resolve_factory(factory)
    source_path = Path(inspect.getsourcefile(factory))
    module = ast.parse(source_path.read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == factory.__name__)
    specialize = Specialize(bindings)
    body = []
    for statement in function.body:
        statement = specialize.visit(statement)
        if isinstance(statement, ast.Assign) and all(isinstance(target, ast.Name) for target in statement.targets):
            try:
                value = constant_value(statement.value)
            except ValueError:
                pass
            else:
                bindings.update({target.id: value for target in statement.targets})
                continue
        body.extend(statement if isinstance(statement, list) else [statement])
    function.name = "make_kernel"
    function.args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    function.body = body
    used = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    imports = []
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias for alias in node.names if (alias.asname or alias.name.split(".")[0]) in used]
            if names:
                node.names = names
                imports.append(node)
    snapshot = ast.Module(body=[*imports, function], type_ignores=[])
    return ast.unparse(ast.fix_missing_locations(snapshot)) + "\n"
