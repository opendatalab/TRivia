import os
import re
from typing import TYPE_CHECKING, Dict, List, Union

import json

if TYPE_CHECKING:
    from swift.llm import InferRequest


class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify==0.5.2'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # edge case
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, soft_max_length, soft_cache_length):
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


# Modified In
import re
import itertools
import html
import unicodedata
import random
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
from bs4 import BeautifulSoup
from collections import deque
from apted import APTED, Config
from apted.helpers import Tree
from Levenshtein import distance
from lxml import etree
import lxml
from lxml.html import HtmlElement
from openai import OpenAI

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

def normalize_html_omni(text):
    def process_table_html(md_i):
        """
        pred_md format edit
        """
        def process_table_html(html_content):
            soup = BeautifulSoup(html_content, 'html.parser')
            th_tags = soup.find_all('th')
            for th in th_tags:
                th.name = 'td'
            thead_tags = soup.find_all('thead')
            for thead in thead_tags:
                thead.unwrap()  # unwrap()会移除标签但保留其内容
            math_tags = soup.find_all('math')
            for math_tag in math_tags:
                alttext = math_tag.get('alttext', '')
                alttext = f'${alttext}$'
                if alttext:
                    math_tag.replace_with(alttext)
            span_tags = soup.find_all('span')
            for span in span_tags:
                span.unwrap()
            return str(soup)

        table_res=''
        table_res_no_space=''
        if '<table' in md_i.replace(" ","").replace("'",'"'):
            md_i = process_table_html(md_i)
            table_res = html.unescape(md_i).replace('\n', '')
            table_res = unicodedata.normalize('NFKC', table_res).strip()
            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, table_res, re.DOTALL | re.IGNORECASE)
            table_res = ''.join(tables)
            # table_res = re.sub('<table.*?>','',table_res)
            table_res = re.sub('( style=".*?")', "", table_res)
            table_res = re.sub('( height=".*?")', "", table_res)
            table_res = re.sub('( width=".*?")', "", table_res)
            table_res = re.sub('( align=".*?")', "", table_res)
            table_res = re.sub('( class=".*?")', "", table_res)
            table_res = re.sub('</?tbody>',"",table_res)
            
            table_res = re.sub(r'\s+', " ", table_res)
            table_res_no_space = '<html><body><table border="1" >' + table_res.replace(' ','') + '</table></body></html>'
            # table_res_no_space = re.sub(' (style=".*?")',"",table_res_no_space)
            # table_res_no_space = re.sub(r'[ ]', " ", table_res_no_space)
            table_res_no_space = re.sub('colspan="', ' colspan="', table_res_no_space)
            table_res_no_space = re.sub('rowspan="', ' rowspan="', table_res_no_space)
            table_res_no_space = re.sub('border="', ' border="', table_res_no_space)

            table_res = '<html><body><table border="1" >' + table_res + '</table></body></html>'
            # table_flow.append(table_res)
            # table_flow_no_space.append(table_res_no_space)

        return table_res, table_res_no_space
    
    def clean_table(input_str,flag=True):
        if flag:
            input_str = input_str.replace('<sup>', '').replace('</sup>', '')
            input_str = input_str.replace('<sub>', '').replace('</sub>', '')
            input_str = input_str.replace('<span>', '').replace('</span>', '')
            input_str = input_str.replace('<div>', '').replace('</div>', '')
            input_str = input_str.replace('<p>', '').replace('</p>', '')
            input_str = input_str.replace('<spandata-span-identity="">', '')
            input_str = re.sub('<colgroup>.*?</colgroup>','',input_str)
        return input_str
    
    norm_text, _ = process_table_html(text)
    norm_text = clean_table(norm_text)
    return norm_text.replace('> ', '>').replace(" </td>", "</td>")

def norm_func(table):
    if "<nl>\n" in table:
        table = convert_otsl_to_html(table)
    return normalize_html_omni(table).replace('<html><body><table border="1" >', "<table>").replace('</table></body></html>', "</table>")


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, span_cell_only=None, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.span_cell_only = span_cell_only
        self.config = CustomConfig if span_cell_only != "strict" else StrictConfig
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = lxml.html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = lxml.html.fromstring(pred, parser=parser)
        true = lxml.html.fromstring(true, parser=parser)
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)

            if self.span_cell_only in ["SC-TT", "SC-TF", "SC-F"]:
                # 裁剪掉colspan和rowspan都为1的节点
                def prune_tree(node):
                    if not node:
                        return None
                    if node.tag == "td" and node.colspan == 1 and node.rowspan == 1:
                        return None
                    node.children = [c for c in node.children if prune_tree(c) is not None]
                    return node

                # tree_pred = prune_tree(tree_pred)
                # tree_true = prune_tree(tree_true)

                # 计算F1分数,考虑结构对应关系
                def get_span_info_with_path(node, path=""):
                    if not node:
                        return {}
                    spans = {}
                    if node.tag == "td":
                        # 将路径信息加入特征中以保持结构对应关系
                        spans[path] = node 
                    for i, child in enumerate(node.children):
                        # 将当前节点的位置信息加入路径
                        child_path = f"{path}/{i}" if path else str(i)
                        spans.update(get_span_info_with_path(child, child_path))
                    return spans
                
                pred_spans = get_span_info_with_path(tree_pred)
                true_spans = get_span_info_with_path(tree_true)
                
                if len(pred_spans) == 0 and len(true_spans) == 0:
                    score = 0.0
                elif len(pred_spans) == 0 and len(true_spans) != 0:
                    score = 0.0
                elif len(pred_spans) != 0 and len(true_spans) == 0:
                    score = -1.0
                else:
                    # 计算合并单元格的F1分数
                    cell_true_span_true = 0
                    cell_true_span_false = 0
                    cell_false = 0
                    # 预测为合并单元格的总数
                    pred_positives = sum(1 for node in pred_spans.values() 
                                       if node.colspan > 1 or node.rowspan > 1)
                    # 真实合并单元格的总数                    
                    true_positives = sum(1 for node in true_spans.values()
                                       if node.colspan > 1 or node.rowspan > 1)
                    
                    # 遍历预测的合并单元格
                    for path, pred_node in pred_spans.items():
                        if pred_node.colspan > 1 or pred_node.rowspan > 1:
                            if path in true_spans:
                                true_node = true_spans[path]
                                # 检查colspan和rowspan是否完全匹配
                                if pred_node.colspan == true_node.colspan and \
                                   pred_node.rowspan == true_node.rowspan:
                                    cell_true_span_true += 1
                                elif true_node.colspan > 1 or true_node.rowspan > 1:
                                    # 预测错误的合并单元格大小，但确实是合并单元格
                                    cell_true_span_false += 0.25
                                else:
                                    # 预测错误的合并单元格，实际并不是合并单元格
                                    cell_false -= 0.75
                            else:
                                # 预测错误的合并单元格，实际并不是合并单元格
                                cell_false -= 0.75
                    if self.span_cell_only == "SC-TT":
                        score = cell_true_span_true / true_positives
                    elif self.span_cell_only == "SC-TF":
                        score = cell_true_span_false / true_positives
                    elif self.span_cell_only == "SC-F":
                        score = cell_false / true_positives
                    else:
                        raise NotImplementedError()
                return score

            distance = APTED(tree_pred, tree_true, self.config()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

    def __call__(self, pred, true):
        return self.evaluate(pred, true)

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        # if self.n_jobs == 1:
        scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        # else:
        #     inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]['html']} for filename in samples]
        #     scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        scores = dict(zip(samples, scores))
        return scores


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


import jieba
from collections import Counter
def f1_en(prediction, ground_truth, normalize_fn):
    if prediction.startswith("ERROR"):
        return 0
    if "not answerable" in prediction.lower():
        return 0
    prediction_tokens = ' '.join(jieba.cut(normalize_fn(prediction)))
    ground_truth_tokens = ' '.join(jieba.cut(normalize_fn(ground_truth)))
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

en_llm_prompt = """Given an HTML-formatted table and a corresponding question, your task is to respond appropriately based on table. If the table do not contain the answer of question, output \"Not answerable\".
Your answer should be a short phrase of only few words. Output the answer within <answer> </answer>.

HTML Table: {}

Question: {}"""

zh_llm_prompt = """给定一个HTML格式的表格以及一个相应的问题，你的任务是根据表格回答该问题。如果该表格不包含该问题的答案，请输出\"无法回答\"。你的答案必须简短、仅有一两个词语。输出答案时用<answer></answer>包裹。

HTML表格: {}

问题: {}"""

def contains_chinese(text):
    """判断文本中是否包含中文字符"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

class OpenAIQAF1Score(ORM):
    def __init__(self) -> None:
        super().__init__()
        self.clients = []
        for url in eval(os.environ["llm_serve_urls"]):
            self.clients.append(OpenAI(
                api_key="EMPTY", base_url=f"{url}/v1"
            ))

    def gen_ans(self, questions, completions):
        """
        使用多线程一次性为所有 (question, completion) 对调用 self.openai.chat.completions，获取调用结果。
        参数:
            questions: 问题列表，长度与completions一致
            completions: HTML表格内容列表
        返回:
            每个 (question, completion) 对应的模型回答列表
        """
        import concurrent.futures

        def call_openai(args):
            question, completion = args
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    if "<nl>" in completion and (not (completion.endswith("<nl>\n") or completion.endswith("<nl>"))):
                        print("Bad results")
                        return "ERROR"
                    html_table = norm_func(completion)
                    if len(html_table) > 30000:
                        html_table = html_table[:30000]
                    client = random.choice(self.clients)
                    prompt = zh_llm_prompt if contains_chinese(question) else en_llm_prompt
                    response = client.chat.completions.create(
                        model="Qwen/Qwen3-8B",
                        messages=[
                            {"role": "user", "content": prompt.format(question, html_table)},
                        ],
                        max_tokens=100,
                        temperature=0,
                        top_p=0.8,
                        presence_penalty=1.5,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt == max_retries - 1:
                        return f"ERROR: {e}"
            return "ERROR"

        # print("start_generating")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(120, len(completions))) as executor:
            futures = {executor.submit(call_openai, item): idx for idx, item in enumerate(zip(questions, completions))}
            ordered_results = [None] * len(futures)
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception as e:
                    raise e
                    ordered_results[idx] = "ERROR"

        return ordered_results
        
    def __call__(self, completions, solution=None, qa_pairs=None, **kwargs):
        """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
        rewards = []
        if qa_pairs is not None:
            # qa_pairs情况：每个completion对应一个qa_pair列表
            all_questions = []
            all_completions = []
            completion_indices = []  # 记录每个问题对应的completion索引
            
            # 展开所有问题和对应的completion
            for comp_idx, (completion, qa_pair) in enumerate(zip(completions, qa_pairs)):
                for qa in qa_pair:
                    all_questions.append(qa["question"])
                    all_completions.append(completion)
                    completion_indices.append(comp_idx)
            
            # 批量生成答案
            all_answers = self.gen_ans(all_questions, all_completions)
            
            # 计算每个completion的平均reward
            completion_rewards = [[] for _ in range(len(completions))]
            completion_qa_strings = [[] for _ in range(len(completions))]  # 存储每个completion的QA字符串
            
            for ans, qa_idx in zip(all_answers, range(len(all_questions))):
                comp_idx = completion_indices[qa_idx]
                qa_pair_idx = qa_idx - sum(len(qa_pairs[i]) for i in range(comp_idx))
                ground_truth = qa_pairs[comp_idx][qa_pair_idx]["answer"]
                question_text = qa_pairs[comp_idx][qa_pair_idx]["question"]
                
                reward = 0.0
                try:
                    # Extract answer from ground truth if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', ground_truth)
                    gt_answer = sol_match.group(1).strip() if sol_match else ground_truth.strip()

                    # Extract answer from content if it has think/answer tags
                    if ans is not None:
                        content_match = re.search(r'<answer>(.*?)</answer>', ans)
                        student_answer = content_match.group(1).strip() if content_match else ans.strip()
                    else:
                        student_answer = ""

                    # Compare the extracted answers
                    reward = f1_en(student_answer, gt_answer, lambda x: x)
                except Exception as e:
                    raise e
                    # pass  # Keep reward as 0.0 if both methods fail

                # 添加QA字符串
                qa_string = f'("{question_text}", "{ans if ans is not None else ""}", "{gt_answer}", "{reward}")'
                completion_qa_strings[comp_idx].append(qa_string)
                
                completion_rewards[comp_idx].append(reward)
            
            # 计算每个completion的平均reward并构建最终的answers
            final_answers = []
            for comp_idx in range(len(completions)):
                if completion_rewards[comp_idx]:
                    avg_reward = sum(completion_rewards[comp_idx]) / len(completion_rewards[comp_idx])
                else:
                    avg_reward = 0.0
                rewards.append(avg_reward)
                
                # 将同一个completion的所有QA对拼接为一个字符串
                final_answer = "\n".join(completion_qa_strings[comp_idx])
                final_answers.append(final_answer)
            
            return (rewards, final_answers)
        else:
            return ([0.0] * len(completions), [""] * len(completions))


class TEDSRewardFunction(ORM):
    def __call__(self, completions, solution, **kwargs):
        """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
        contents = completions
        rewards = []
        for content, sol in zip(contents, solution):
            reward = 0.0

            # If symbolic verification failed, try string matching
            if len(content) > 30000:
                content = content[:30000]
            if len(sol) > 30000:
                sol = sol[:30000]
            if sol == "<table></table>":
                reward = 0.0
            elif reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    reward = TEDS()(norm_func(student_answer), norm_func(ground_truth))
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail

            rewards.append(reward)
        return rewards

orms = {
    'QA_F1_score': OpenAIQAF1Score,
    'TEDS': TEDSRewardFunction,
# Modified Out
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
}
