from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import html as html_lib
from html.parser import HTMLParser
import itertools
import re
import unicodedata


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


@dataclass
class TableCell:
    text: str
    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    row_span: int = 1
    col_span: int = 1
    column_header: bool = False


@dataclass
class TableData:
    table_cells: list[TableCell]
    num_rows: int = 0
    num_cols: int = 0

    @property
    def grid(self) -> list[list[TableCell]]:
        table_data = [
            [
                TableCell(
                    text="",
                    start_row_offset_idx=row_index,
                    end_row_offset_idx=row_index + 1,
                    start_col_offset_idx=col_index,
                    end_col_offset_idx=col_index + 1,
                )
                for col_index in range(self.num_cols)
            ]
            for row_index in range(self.num_rows)
        ]

        for cell in self.table_cells:
            for row_index in range(
                min(cell.start_row_offset_idx, self.num_rows),
                min(cell.end_row_offset_idx, self.num_rows),
            ):
                for col_index in range(
                    min(cell.start_col_offset_idx, self.num_cols),
                    min(cell.end_col_offset_idx, self.num_cols),
                ):
                    table_data[row_index][col_index] = cell

        return table_data


OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"
OTSL_TOKENS = (OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL)


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


class TableMarkupNormalizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.math_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "math":
            alttext = dict(attrs).get("alttext", "")
            self.parts.append(f"${alttext}$")
            self.math_depth += 1
            return
        if self.math_depth:
            return
        if tag in {"thead", "span"}:
            return

        output_tag = "td" if tag == "th" else tag
        self.parts.append(f"<{output_tag}{self._format_attrs(attrs)}>")

    def handle_endtag(self, tag: str) -> None:
        if tag == "math":
            self.math_depth -= 1
            return
        if self.math_depth or tag in {"thead", "span"}:
            return

        output_tag = "td" if tag == "th" else tag
        self.parts.append(f"</{output_tag}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "math":
            alttext = dict(attrs).get("alttext", "")
            self.parts.append(f"${alttext}$")
            return
        if self.math_depth or tag in {"thead", "span"}:
            return

        output_tag = "td" if tag == "th" else tag
        self.parts.append(f"<{output_tag}{self._format_attrs(attrs)}/>")

    def handle_data(self, data: str) -> None:
        if not self.math_depth:
            self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if not self.math_depth:
            self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if not self.math_depth:
            self.parts.append(f"&#{name};")

    def normalized(self) -> str:
        return "".join(self.parts)

    def _format_attrs(self, attrs: list[tuple[str, str | None]]) -> str:
        return "".join(
            f' {name}="{html_lib.escape(value, quote=True)}"' if value is not None else f" {name}"
            for name, value in attrs
        )


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


def has_table(html: str) -> bool:
    return "<table" in html.lower()


def otsl_extract_tokens_and_text(text: str) -> tuple[list[str], list[str]]:
    pattern = r"(" + r"|".join(OTSL_TOKENS) + r")"
    tokens = re.findall(pattern, text)
    text_parts = [part for part in re.split(pattern, text) if part.strip()]
    return tokens, text_parts


def otsl_parse_texts(texts: list[str], tokens: list[str]) -> tuple[list[TableCell], list[list[str]]]:
    split_row_tokens = [
        list(row_tokens)
        for is_separator, row_tokens in itertools.groupby(tokens, lambda token: token == OTSL_NL)
        if not is_separator
    ]
    table_cells = []
    row_index = 0
    col_index = 0

    if split_row_tokens:
        max_cols = max(len(row) for row in split_row_tokens)

        for row in split_row_tokens:
            while len(row) < max_cols:
                row.append(OTSL_ECEL)

        new_texts = []
        text_index = 0

        for row in split_row_tokens:
            for token in row:
                new_texts.append(token)
                if text_index < len(texts) and texts[text_index] == token:
                    text_index += 1
                    if text_index < len(texts) and texts[text_index] not in OTSL_TOKENS:
                        new_texts.append(texts[text_index])
                        text_index += 1

            new_texts.append(OTSL_NL)
            if text_index < len(texts) and texts[text_index] == OTSL_NL:
                text_index += 1

        texts = new_texts

    def count_right(row_tokens: list[list[str]], col_index: int, row_index: int, which_tokens: list[str]) -> int:
        span = 0
        col_iter = col_index
        while row_tokens[row_index][col_iter] in which_tokens:
            col_iter += 1
            span += 1
            if col_iter >= len(row_tokens[row_index]):
                return span
        return span

    def count_down(row_tokens: list[list[str]], col_index: int, row_index: int, which_tokens: list[str]) -> int:
        span = 0
        row_iter = row_index
        while row_tokens[row_iter][col_index] in which_tokens:
            row_iter += 1
            span += 1
            if row_iter >= len(row_tokens):
                return span
        return span

    for index, text in enumerate(texts):
        cell_text = ""
        if text in [OTSL_FCEL, OTSL_ECEL]:
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != OTSL_ECEL and texts[index + 1] not in OTSL_TOKENS:
                cell_text = texts[index + 1]
                right_offset = 2

            next_right_cell = ""
            if index + right_offset < len(texts):
                next_right_cell = texts[index + right_offset]

            next_bottom_cell = ""
            if row_index + 1 < len(split_row_tokens) and col_index < len(split_row_tokens[row_index + 1]):
                next_bottom_cell = split_row_tokens[row_index + 1][col_index]

            if next_right_cell in [OTSL_LCEL, OTSL_XCEL]:
                col_span += count_right(split_row_tokens, col_index + 1, row_index, [OTSL_LCEL, OTSL_XCEL])
            if next_bottom_cell in [OTSL_UCEL, OTSL_XCEL]:
                row_span += count_down(split_row_tokens, col_index, row_index + 1, [OTSL_UCEL, OTSL_XCEL])

            table_cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_index,
                    end_row_offset_idx=row_index + row_span,
                    start_col_offset_idx=col_index,
                    end_col_offset_idx=col_index + col_span,
                )
            )
        if text in OTSL_TOKENS[1:]:
            col_index += 1
        if text == OTSL_NL:
            row_index += 1
            col_index = 0

    return table_cells, split_row_tokens


def export_otsl_to_html(table_data: TableData) -> str:
    if not table_data.table_cells:
        return ""

    current_grid = table_data.grid
    html_parts = []
    for row_index in range(table_data.num_rows):
        html_parts.append("<tr>")
        for col_index in range(table_data.num_cols):
            cell = current_grid[row_index][col_index]

            if cell.start_row_offset_idx != row_index or cell.start_col_offset_idx != col_index:
                continue

            opening_tag_parts = ["<th" if cell.column_header else "<td"]
            if cell.row_span > 1:
                opening_tag_parts.append(f' rowspan="{cell.row_span}"')
            if cell.col_span > 1:
                opening_tag_parts.append(f' colspan="{cell.col_span}"')
            opening_tag_parts.append(">")
            cell_tag_name = "th" if cell.column_header else "td"
            html_parts.append(f"{''.join(opening_tag_parts)}{cell.text.strip()}</{cell_tag_name}>")
        html_parts.append("</tr>")

    return f"<table>{''.join(html_parts)}</table>"


def convert_otsl_to_html(otsl_content: str) -> str:
    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)
    table_data = TableData(
        num_rows=len(split_row_tokens),
        num_cols=max(len(row) for row in split_row_tokens) if split_row_tokens else 0,
        table_cells=table_cells,
    )
    return export_otsl_to_html(table_data)


def normalize_html_omni(text: str) -> str:
    def process_table_html(html_content: str) -> str:
        parser = TableMarkupNormalizer()
        parser.feed(html_content)
        return parser.normalized()

    def clean_table(input_str: str) -> str:
        input_str = input_str.replace("<sup>", "").replace("</sup>", "")
        input_str = input_str.replace("<sub>", "").replace("</sub>", "")
        input_str = input_str.replace("<span>", "").replace("</span>", "")
        input_str = input_str.replace("<div>", "").replace("</div>", "")
        input_str = input_str.replace("<p>", "").replace("</p>", "")
        input_str = input_str.replace('<spandata-span-identity="">', "")
        return re.sub("<colgroup>.*?</colgroup>", "", input_str)

    table_res = ""
    if "<table" in text.replace(" ", "").replace("'", '"'):
        table_res = html_lib.unescape(process_table_html(text)).replace("\n", "")
        table_res = unicodedata.normalize("NFKC", table_res).strip()
        tables = re.findall(r"<table\b[^>]*>(.*)</table>", table_res, re.DOTALL | re.IGNORECASE)
        table_res = "".join(tables)
        table_res = re.sub('( style=".*?")', "", table_res)
        table_res = re.sub('( height=".*?")', "", table_res)
        table_res = re.sub('( width=".*?")', "", table_res)
        table_res = re.sub('( align=".*?")', "", table_res)
        table_res = re.sub('( class=".*?")', "", table_res)
        table_res = re.sub("</?tbody>", "", table_res)
        table_res = re.sub(r"\s+", " ", table_res)
        table_res = '<html><body><table border="1" >' + table_res + "</table></body></html>"

    norm_text = clean_table(table_res)
    return norm_text.replace("> ", ">").replace(" </td>", "</td>")


def normalize_table_text(table: str) -> str:
    if "<nl>\n" in table:
        table = convert_otsl_to_html(table)
    return normalize_html_omni(table).replace(
        '<html><body><table border="1" >',
        "<table>",
    ).replace("</table></body></html>", "</table>")


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
    if not has_table(prediction) or not has_table(reference):
        return 0.0
    pred_tree = html_to_tree(prediction)
    ref_tree = html_to_tree(reference)
    normalizer = max(pred_tree.size, ref_tree.size)
    return 1.0 - tree_edit_distance(pred_tree, ref_tree) / normalizer

