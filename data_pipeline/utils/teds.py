from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser


@dataclass(frozen=True)
class TableNode:
    label: str
    text: str
    children: tuple["TableNode", ...]

    @property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)


@dataclass
class MutableTableNode:
    tag: str
    attrs: dict[str, str]
    text_parts: list[str]
    children: list["MutableTableNode"]


class TableHTMLParser(HTMLParser):
    table_tags = {"table", "thead", "tbody", "tfoot", "tr", "td", "th"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root: MutableTableNode | None = None
        self.stack: list[MutableTableNode] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self.table_tags:
            return
        node = MutableTableNode(
            tag=tag,
            attrs={key: value or "" for key, value in attrs},
            text_parts=[],
            children=[],
        )
        if self.stack:
            self.stack[-1].children.append(node)
        if tag == "table" and self.root is None:
            self.root = node
        self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.table_tags:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self.stack and self.stack[-1].tag in {"td", "th"}:
            self.stack[-1].text_parts.append(data)


def _label(node: MutableTableNode) -> str:
    if node.tag in {"td", "th"}:
        colspan = node.attrs.get("colspan", "1")
        rowspan = node.attrs.get("rowspan", "1")
        return f"{node.tag}:c{colspan}:r{rowspan}"
    return node.tag


def _freeze(node: MutableTableNode) -> TableNode:
    text = " ".join(" ".join(node.text_parts).split()) if node.tag in {"td", "th"} else ""
    return TableNode(
        label=_label(node),
        text=text,
        children=tuple(_freeze(child) for child in node.children),
    )


def html_to_tree(html: str) -> TableNode:
    parser = TableHTMLParser()
    parser.feed(html)
    return _freeze(parser.root)


def _update_cost(left: TableNode, right: TableNode) -> float:
    if left.label != right.label:
        return 1.0
    if left.text or right.text:
        return 1.0 - SequenceMatcher(None, left.text, right.text).ratio()
    return 0.0


def tree_edit_distance(left: TableNode, right: TableNode) -> float:
    rows = len(left.children) + 1
    cols = len(right.children) + 1
    dp = [[0.0 for _ in range(cols)] for _ in range(rows)]

    for row in range(1, rows):
        dp[row][0] = dp[row - 1][0] + left.children[row - 1].size
    for col in range(1, cols):
        dp[0][col] = dp[0][col - 1] + right.children[col - 1].size

    for row in range(1, rows):
        for col in range(1, cols):
            delete_cost = dp[row - 1][col] + left.children[row - 1].size
            insert_cost = dp[row][col - 1] + right.children[col - 1].size
            update_cost = dp[row - 1][col - 1] + tree_edit_distance(
                left.children[row - 1],
                right.children[col - 1],
            )
            dp[row][col] = min(delete_cost, insert_cost, update_cost)

    return _update_cost(left, right) + dp[-1][-1]


def teds_score(prediction: str, reference: str) -> float:
    pred_tree = html_to_tree(prediction)
    ref_tree = html_to_tree(reference)
    normalizer = max(pred_tree.size, ref_tree.size)
    return 1.0 - tree_edit_distance(pred_tree, ref_tree) / normalizer

