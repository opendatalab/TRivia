import re
import itertools
import html
from typing import Any, Dict, Final, List, Literal, Optional, Tuple, Union
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

class TableCell(BaseModel):
    """TableCell."""
    row_span: int = 1
    col_span: int = 1
    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    text: str
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False

    @model_validator(mode="before")
    @classmethod
    def from_dict_format(cls, data: Any) -> Any:
        """from_dict_format."""
        if isinstance(data, Dict):
            # Check if this is a native BoundingBox or a bbox from docling-ibm-models
            if (
                # "bbox" not in data
                # or data["bbox"] is None
                # or isinstance(data["bbox"], BoundingBox)
                "text"
                in data
            ):
                return data
            text = data["bbox"].get("token", "")
            if not len(text):
                text_cells = data.pop("text_cell_bboxes", None)
                if text_cells:
                    for el in text_cells:
                        text += el["token"] + " "

                text = text.strip()
            data["text"] = text

        return data


class TableData(BaseModel):  # TBD
    """BaseTableData."""

    table_cells: List[TableCell] = []
    num_rows: int = 0
    num_cols: int = 0

    @computed_field  # type: ignore
    @property
    def grid(
        self,
    ) -> List[List[TableCell]]:
        """grid."""
        # Initialise empty table data grid (only empty cells)
        table_data = [
            [
                TableCell(
                    text="",
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                for j in range(self.num_cols)
            ]
            for i in range(self.num_rows)
        ]

        # Overwrite cells in table data for which there is actual cell content.
        for cell in self.table_cells:
            for i in range(
                min(cell.start_row_offset_idx, self.num_rows),
                min(cell.end_row_offset_idx, self.num_rows),
            ):
                for j in range(
                    min(cell.start_col_offset_idx, self.num_cols),
                    min(cell.end_col_offset_idx, self.num_cols),
                ):
                    table_data[i][j] = cell

        return table_data

"""
OTSL
"""
OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"

def otsl_extract_tokens_and_text(s: str):
    # Pattern to match anything enclosed by < >
    # (including the angle brackets themselves)
    # pattern = r"(<[^>]+>)"
    pattern = r"(" + r"|".join([OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]) + r")"
    # Find all tokens (e.g. "<otsl>", "<loc_140>", etc.)
    tokens = re.findall(pattern, s)
    # Remove any tokens that start with "<loc_"
    tokens = [token for token in tokens]
    # Split the string by those tokens to get the in-between text
    text_parts = re.split(pattern, s)
    text_parts = [token for token in text_parts]
    # Remove any empty or purely whitespace strings from text_parts
    text_parts = [part for part in text_parts if part.strip()]

    return tokens, text_parts

def otsl_parse_texts(texts, tokens):
    split_word = OTSL_NL
    split_row_tokens = [
        list(y)
        for x, y in itertools.groupby(tokens, lambda z: z == split_word)
        if not x
    ]
    table_cells = []
    r_idx = 0
    c_idx = 0

    # 检查并补充矩阵以使其完整
    if split_row_tokens:
        # 找到最大列数
        max_cols = max(len(row) for row in split_row_tokens)
        
        # 补充每一行使其达到最大列数
        for row_idx, row in enumerate(split_row_tokens):
            while len(row) < max_cols:
                row.append(OTSL_ECEL)
        
        # 在texts中也需要相应补充<ecel>
        # 重新构建texts以包含补充的<ecel>
        new_texts = []
        text_idx = 0
        
        for row_idx, row in enumerate(split_row_tokens):
            for col_idx, token in enumerate(row):
                new_texts.append(token)
                # 如果这个token在原始texts中有对应的文本内容，添加它
                if text_idx < len(texts) and texts[text_idx] == token:
                    text_idx += 1
                    # 检查下一个是否是文本内容（不是token）
                    if (text_idx < len(texts) and 
                        texts[text_idx] not in [OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]):
                        new_texts.append(texts[text_idx])
                        text_idx += 1

            new_texts.append(OTSL_NL)
            if text_idx < len(texts) and texts[text_idx] == OTSL_NL:
                text_idx += 1
        
        texts = new_texts

    def count_right(tokens, c_idx, r_idx, which_tokens):
        span = 0
        c_idx_iter = c_idx
        while tokens[r_idx][c_idx_iter] in which_tokens:
            c_idx_iter += 1
            span += 1
            if c_idx_iter >= len(tokens[r_idx]):
                return span
        return span

    def count_down(tokens, c_idx, r_idx, which_tokens):
        span = 0
        r_idx_iter = r_idx
        while tokens[r_idx_iter][c_idx] in which_tokens:
            r_idx_iter += 1
            span += 1
            if r_idx_iter >= len(tokens):
                return span
        return span

    for i, text in enumerate(texts):
        cell_text = ""
        if text in [
            OTSL_FCEL,
            OTSL_ECEL,
        ]:
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != OTSL_ECEL and (texts[i + 1] not in [OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]):
                cell_text = texts[i + 1]
                right_offset = 2

            # Check next element(s) for lcel / ucel / xcel,
            # set properly row_span, col_span
            next_right_cell = ""
            if i + right_offset < len(texts):
                next_right_cell = texts[i + right_offset]

            next_bottom_cell = ""
            if r_idx + 1 < len(split_row_tokens):
                if c_idx < len(split_row_tokens[r_idx + 1]):
                    next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]

            if next_right_cell in [
                OTSL_LCEL,
                OTSL_XCEL,
            ]:
                # we have horisontal spanning cell or 2d spanning cell
                col_span += count_right(
                    split_row_tokens,
                    c_idx + 1,
                    r_idx,
                    [OTSL_LCEL, OTSL_XCEL],
                )
            if next_bottom_cell in [
                OTSL_UCEL,
                OTSL_XCEL,
            ]:
                # we have a vertical spanning cell or 2d spanning cell
                row_span += count_down(
                    split_row_tokens,
                    c_idx,
                    r_idx + 1,
                    [OTSL_UCEL, OTSL_XCEL],
                )

            table_cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                )
            )
        if text in [
            OTSL_FCEL,
            OTSL_ECEL,
            OTSL_LCEL,
            OTSL_UCEL,
            OTSL_XCEL,
        ]:
            c_idx += 1
        if text == OTSL_NL:
            r_idx += 1
            c_idx = 0
    return table_cells, split_row_tokens

# def export_to_html(table_data: TableData):
#     nrows = table_data.num_rows
#     ncols = table_data.num_cols
#     if len(table_data.table_cells) == 0:
#         return ""

#     body = ""
#     grid = table_data.grid

#     for i in range(nrows):
#         body += "<tr>"
#         for j in range(ncols):
#             cell: TableCell = grid[i][j]

#             rowspan, rowstart = (
#                 cell.row_span,
#                 cell.start_row_offset_idx,
#             )
#             colspan, colstart = (
#                 cell.col_span,
#                 cell.start_col_offset_idx,
#             )

#             if rowstart != i:
#                 continue
#             if colstart != j:
#                 continue

#             content = html.escape(cell.text.strip())
#             celltag = "td"
#             if cell.column_header:
#                 celltag = "th"

#             opening_tag = f"{celltag}"
#             if rowspan > 1:
#                 opening_tag += f' rowspan="{rowspan}"'
#             if colspan > 1:
#                 opening_tag += f' colspan="{colspan}"'

#             body += f"<{opening_tag}>{content}</{celltag}>"
#         body += "</tr>"

#     # dir = get_text_direction(text)
#     body = f"<table>{body}</table>"

#     return body

def export_to_html(table_data: TableData) -> str:
    nrows = table_data.num_rows
    ncols = table_data.num_cols
    # print(nrows, ncols)

    if not table_data.table_cells:
        return ""

    current_grid = table_data.grid

    html_str_list = []

    for i in range(nrows):
        html_str_list.append("<tr>")
        for j in range(ncols):
            cell: TableCell = current_grid[i][j]

            if cell.start_row_offset_idx != i or cell.start_col_offset_idx != j:
                continue

            # content = html.escape(cell.text.strip())
            content = cell.text.strip()
            cell_tag_name = "th" if cell.column_header else "td"

            opening_tag_parts = [f"<{cell_tag_name}"]
            if cell.row_span > 1:
                opening_tag_parts.append(f' rowspan="{cell.row_span}"')
            if cell.col_span > 1:
                opening_tag_parts.append(f' colspan="{cell.col_span}"')
            opening_tag_parts.append(">")
            opening_tag = "".join(opening_tag_parts)

            html_str_list.append(f"{opening_tag}{content}</{cell_tag_name}>")
        html_str_list.append("</tr>")

    body_content = "".join(html_str_list)
    return f"<table>{body_content}</table>"

def convert_otsl_to_html(otsl_content: str) -> str:
    # if not otsl_content.endswith("<nl>\n"):
    #     return ""

    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)

    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)

    table_data = TableData(
                num_rows=len(split_row_tokens),
                num_cols=(
                    max(len(row) for row in split_row_tokens) if split_row_tokens else 0
                ),
                table_cells=table_cells,
            )

    result = export_to_html(table_data)
    
    return result

if __name__ == "__main__":
    import time
    
    # test
    a = """
    <fcel><nl>\n
    <fcel><nl>\n"""
    b = """<fcel>Reviewer<fcel>Representation<fcel>Consultant<fcel>Speaker's Bureau<fcel>Ownership/ Partnership/ Principal<fcel>Personal Research<fcel>Institutional, Organizational, or Other Financial Benefit<fcel>Expert Witness<nl>
<fcel>John E. Brush<fcel>Official Reviewer–ACCF Board of Trustees<fcel>● United Healthcare<fcel>None<fcel>None<fcel>None<fcel>● PROMETHEUS Payment (Board member)<fcel>None<nl>
<fcel>David P. Faxon<fcel>Official Reviewer–AHA<fcel>● Johnson & Johnson<fcel>None<fcel>● CULPRIT Trial (PI)*<fcel>None<fcel>● Circulation: Cardiovascular Interventions—Editor*<fcel>None<nl>
<ucel><ucel><ucel><ucel><fcel>● RIVA Medical<ucel><ucel><ucel><nl>
<fcel>Robert A. Harrington<fcel>Official Reviewer–AHA<fcel>● AstraZeneca*<fcel>None<fcel>None<fcel>● AstraZeneca<fcel>None<fcel>None<nl>
<ucel><ucel><fcel>● Baxter<ucel><ucel><fcel>● Baxter<ucel><ucel><nl>
<ucel><ucel><fcel>● CSL Behring<ucel><ucel><fcel>● Bristol-Myers Squibb*<ucel><ucel><nl>
<ucel><ucel><fcel>● Eli Lilly<ucel><ucel><fcel>● GlaxoSmithKline<ucel><ucel><nl>
<ucel><ucel><fcel>● Luiypold<ucel><ucel><fcel>● The Medicines Company<ucel><ucel><nl>
<ucel><ucel><fcel>● Merck<ucel><ucel><fcel>● Merck*<ucel><ucel><nl>
<ucel><ucel><fcel>● Novartis<ucel><ucel><fcel>● Portola*<ucel><ucel><nl>
<ucel><ucel><fcel>● Otsuka Maryland Research Institute<ucel><ucel><fcel>● Schering-Plough*<ucel><ucel><nl>
<ucel><ucel><fcel>● Regado<ucel><ucel><ucel><ucel><ucel><nl>
<ucel><ucel><fcel>● Sanofi-aventis<ucel><ucel><ucel><ucel><ucel><nl>
<ucel><ucel><fcel>● Schering-Plough*<ucel><ucel><ucel><ucel><ucel><nl>
<ucel><ucel><fcel>● WebMD*<ucel><ucel><ucel><ucel><ucel><nl>
<fcel>Judith S. Hochman<fcel>Official Reviewer–ACCF/AHA Task Force on Practice Guidelines<fcel>● BMS/Sanofi<fcel>None<fcel>None<fcel>● Johnson & Johnson/Bayer Healthcare AG (DSMB)<fcel>None<fcel>None<nl>
<ucel><ucel><fcel>● Eli Lilly<ucel><ucel><ucel><ucel><ucel><nl>
<ucel><ucel><fcel>● GlaxoSmithKline<ucel><ucel><ucel><ucel><ucel><nl>
<ucel><ucel><fcel>● Millennium Pharmaceuticals/ Schering-Plough<ucel><ucel><fcel>● Schering-Plough (TIMI 50) (DSMB)<ucel><ucel><nl>
<fcel>Rodney H. Zimmermann<fcel>Official Reviewer–ACCF Board of Governors<fcel>● AstraZeneca<fcel>● AstraZeneca<fcel>None<fcel>● AstraZeneca<fcel>None<fcel>None<nl>
<ucel><ucel><fcel>● Boehringer Ingelheim<fcel>● Merck-Frost<fcel>● Sanofi-aventis<ucel><fcel>● Sanofi-aventis<ucel><nl>
<ucel><ucel><fcel>● Bristol-Myers Squibb<fcel>● Servier<ucel><ucel><ucel><ucel><nl>
<ucel><ucel><fcel>● Medtronic<ucel><ucel><ucel><ucel><ucel><nl>
<ucel><ucel><fcel>● Sanofi-aventis<ucel><ucel><ucel><ucel><ucel><nl>
<ucel><ucel><fcel>● Schering-Plough<ucel><ucel><ucel><ucel><ucel><nl>
<fcel>Steven Brown<fcel>Organizational Reviewer–AAFP<fcel>None<fcel>None<fcel>None<fcel>None<fcel>None<fcel>None<nl>
<fcel>Joseph C. Cleveland<fcel>Organizational Reviewer–STS<fcel>● Baxter Biosurgery<fcel>None<fcel>None<fcel>None<fcel>● Heartware<fcel>None<nl>
<ucel><ucel><fcel>● Essential Pharmaceuticals<ucel><ucel><ucel><fcel>● Thoratec<ucel><nl>
<fcel>Wyatt Decker<fcel>Organizational Reviewer–ACEP<fcel>None<fcel>None<fcel>None<fcel>None<fcel>None<fcel>None<nl>
<fcel>Joseph A. de Gregorio<fcel>Organizational Reviewer–SCAI<fcel>None<fcel>None<fcel>None<fcel>None<fcel>None<fcel>None<nl>
<fcel>Deborah B. Diercks<fcel>Organizational Reviewer–ACEP<fcel>● AstraZeneca<fcel>None<fcel>None<fcel>None<fcel>● Society of Chest Pain Centers and Providers<fcel>None<nl>
<ucel><ucel><fcel>● Sanofi-aventis<ucel><ucel><ucel><ucel><ucel><nl>
<ucel><ucel><fcel>● Schering-Plough<ucel><ucel><ucel><ucel><ucel><nl>
<fcel>Benjamin Hatten<fcel>Organizational Reviewer–ACEP<fcel>None<fcel>None<fcel>None<fcel>None<fcel>None<fcel>None<nl>
<fcel>Loren F. Hiratzka<fcel>Organizational Reviewer–STS<fcel>None<fcel>None<fcel>None<fcel>None<fcel>● Cardiac, Vascular, and Thoracic Surgeons*<fcel>None<nl>
<ucel><ucel><ucel><ucel><ucel><ucel><fcel>● TriHealth (Bethesda North and Good Samaritan Hospitals)*<ucel><nl>
<fcel>Jason H. Rogers<fcel>Organizational Reviewer–SCAI<fcel>● Ample Medical<fcel>None<fcel>None<fcel>None<fcel>None<fcel>None<nl>
<fcel>Vincenza T. Show<fcel>Organizational Reviewer–ACP<fcel>None<fcel>None<fcel>None<fcel>● Boehringer Ingelheim*<fcel>● ACP*<fcel>None<nl>
<ucel><ucel><ucel><ucel><ucel><fcel>● Bristol-Myers Squibb*<ucel><ucel><nl>
"""
    print(convert_otsl_to_html(b))
