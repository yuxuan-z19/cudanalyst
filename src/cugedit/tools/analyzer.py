import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import override

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from ..utils import render_feedback_md
from .base import BaseTool, ToolContext


@dataclass
class Element:
    name: str
    body: str


@dataclass
class Param(Element):
    base_type: str
    ptr_level: int = 0
    array_dims: list[int] = field(default_factory=list)
    qualifiers: list[str] = field(default_factory=list)

    @property
    def dtype(self) -> str:
        return f"{self.base_type}{'*' * self.ptr_level}"

    @property
    def rank(self) -> int:
        return self.ptr_level + len(self.array_dims)


@dataclass
class Loop(Element):
    name: str = field(init=False, default="")
    directive: list[str] = field(default_factory=list)
    children: list["Loop"] = field(default_factory=list)
    level: int = 0


@dataclass
class Function(Element):
    return_type: str
    params: list[Param] = field(default_factory=list)
    qualifiers: list[str] = field(default_factory=list)
    loops: list[Loop] = field(default_factory=list)


class TypeResolver:
    @staticmethod
    @lru_cache
    def _resolve(node: Node) -> str:
        if node.type == "qualified_identifier":
            return "::".join(
                TypeResolver._resolve(c) for c in node.children if c.type != "::"
            )
        elif node.is_named:
            return node.text.decode()
        return ""

    @staticmethod
    def parse(node: Node) -> str:
        return " ".join(
            TypeResolver._resolve(c)
            for c in node.children
            if c.type
            in {
                "primitive_type",
                "type_identifier",
                "qualified_identifier",
                "namespace_identifier",
            }
        )


class ParamParser:
    @staticmethod
    def _parse_declarator(node: Node):
        ptr_level = 0
        array_dims = []
        qualifiers = []
        name = ""

        while node:
            t = node.type
            if t == "pointer_declarator":
                ptr_level += 1
                qualifiers.extend(
                    c.text.decode() for c in node.children if c.type == "type_qualifier"
                )
                node = node.child_by_field_name("declarator")

            elif t == "reference_declarator":
                ptr_level -= 1
                node = node.child_by_field_name("declarator")

            elif t == "array_declarator":
                size = node.child_by_field_name("size")
                array_dims.append(int(size.text.decode()) if size else -1)
                node = node.child_by_field_name("declarator")

            elif t == "parenthesized_declarator":
                node = node.child_by_field_name("declarator")

            elif t == "identifier":
                name = node.text.decode()
                break

            else:
                id_node = next(
                    (c for c in node.children if c.type == "identifier"), None
                )
                name = id_node.text.decode() if id_node else ""
                break

        array_dims.reverse()
        return name, ptr_level, array_dims, qualifiers

    @classmethod
    def parse(cls, param_list_node: Node | None):
        if param_list_node is None:
            return []

        result = []

        for param in param_list_node.children:
            if param.type != "parameter_declaration":
                continue

            base_type = TypeResolver.parse(param)
            declarator_node = param.child_by_field_name("declarator")

            name, ptr_level, array_dims, quals_decl = cls._parse_declarator(
                declarator_node
            )
            quals_total = quals_decl + [
                c.text.decode() for c in param.children if c.type == "type_qualifier"
            ]

            full_decl = name + "".join(f"[{d}]" for d in array_dims)

            result.append(
                Param(
                    name=name,
                    body=full_decl,
                    base_type=base_type,
                    ptr_level=ptr_level,
                    array_dims=array_dims,
                    qualifiers=quals_total,
                )
            )

        return result


class LoopParser:
    @staticmethod
    def _find_directives(node: Node) -> list[str]:
        cur = node.prev_sibling
        directives = []

        while cur:
            if not cur.is_named:
                cur = cur.prev_sibling
                continue
            if cur.type == "preproc_call":
                directives.append(cur.text.decode().strip())
            elif cur.type == "comment":
                cur = cur.prev_sibling
                continue
            else:
                break
            cur = cur.prev_sibling

        return directives[::-1]

    @classmethod
    def _calc_levels(cls, loop: Loop):
        max_child_level = max(
            (cls._calc_levels(child) for child in loop.children), default=0
        )
        loop.level = max_child_level + 1
        return loop.level

    @classmethod
    def parse(cls, node: Node | None) -> list[Loop]:
        if node is None:
            return []

        loop_list = []
        stack = [(node, loop_list)]

        while stack:
            cur_node, loops_target = stack.pop()

            for child in reversed(cur_node.children):
                if child.type == "for_statement":
                    directive = cls._find_directives(child)
                    body = child.text.decode()

                    loop = Loop(directive=directive, body=body, children=[])
                    loops_target.append(loop)

                    body_node = child.child_by_field_name("body")
                    if body_node:
                        stack.append((body_node, loop.children))
                else:
                    stack.append((child, loops_target))

        for loop in loop_list:
            cls._calc_levels(loop)

        return loop_list


class FunctionParser:
    @classmethod
    def parse(cls, func_node: Node) -> Function | None:
        if func_node.type != "function_definition":
            return None

        return_node = func_node.child_by_field_name("type")
        decl_node = func_node.child_by_field_name("declarator")
        body_node = func_node.child_by_field_name("body")

        return_type = return_node.text.decode() if return_node else ""

        name = ""
        params = []

        if decl_node and decl_node.type == "function_declarator":
            inner_decl = decl_node.child_by_field_name("declarator")
            name = inner_decl.text.decode() if inner_decl else ""

            params = ParamParser.parse(decl_node.child_by_field_name("parameters"))

        if not name:
            return None

        body = body_node.text.decode() if body_node else ""
        loops = LoopParser.parse(body_node)

        qualifiers = [
            c.text.decode() for c in func_node.children if c.text.decode() in c.type
        ]

        return Function(
            name=name,
            body=body,
            return_type=return_type,
            params=params,
            qualifiers=qualifiers,
            loops=loops,
        )


class CodeAnlzTool(BaseTool):
    PARSER = get_parser("cuda")

    @override
    @classmethod
    def extract(cls, functions: list[Function]):
        return {
            f.name: [{"code": l.body, "nested level": l.level} for l in f.loops]
            for f in functions
        }

    @override
    @classmethod
    def render(cls, feedback):
        return render_feedback_md(feedback)

    @override
    @classmethod
    def run(cls, ctx: ToolContext):
        code = Path(ctx.code_path).read_text()

        functions = []

        @lru_cache
        def _traverse(n: Node):
            if n.type == "function_definition":
                func = FunctionParser.parse(n)
                if func:
                    functions.append(func)
            else:
                for c in n.children:
                    _traverse(c)

        tree = cls.PARSER.parse(code.encode())
        _traverse(tree.root_node)
        return cls.extract(functions)
