<div align="center">

[English](./README.md) | 简体中文

</div>

<h1 align="center">
TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition
</h1>

<p align="center">
 <img src="./assets/performance.jpg" width="100%"/> <br>
</p>

<p align="center">
<a href=""><b>📜 论文</b></a> |
<a href="https://github.com/opendatalab/TRivia"><b>Github</b></a> |
<a href="https://huggingface.co/spaces/opendatalab/TRivia-3B"><b>🤗 演示(Huggingface)</b></a>
<a href="https://huggingface.co/Carkham/TRivia"><b>🤗 模型权重(Huggingface)</b></a>
</p>

TRivia是一个新颖的自监督表格识别VLM的微调框架。我们在这个仓库中发布了TRivia-3B。TRivia-3B是一个基于Qwen2.5-VL-3B，使用TRivia框架进行微调的先进表格识别VLM，并在多个真实世界的表格识别基准上展现出强大的性能。

# 最新动态
- 2026.07.14 我们发布了 TRivia 的 [QA 监督生成代码](./data_pipeline)。
- 🎉🎉🎉 2026.02.26 TRivia 已被 CVPR 2026 接收。
- 2025.12.26 我们发布了 TRivia 的[训练代码](./training)。

# 关键特性:
- ⭐ 强大的表格识别能力，TRivia-3B不仅适用于电子、扫描和拍照等等表格，而且能自动分辨表格图片中的背景与主体，仅识别表格主体部分。
- 📃 可复现的训练管线，仅使用无标签数据且无需蒸馏即可推动表格识别能力的提升。

<p align="center">
 <img src="./assets/pipeline.jpg" width="100%"/> <br>
</p>

# 基准性能
我们主要在下面三个真实世界基准上进行评测: [OmnidocBench v1.5](https://github.com/opendatalab/OmniDocBench), [CC-OCR](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/Benchmarks/CC-OCR) and [OCRBench v2](https://github.com/Yuliang-Liu/MultimodalOCR)

<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="2">PubTabNet</th>
      <th colspan="2">OmniDocBench</th>
      <th colspan="2">CC-OCR</th>
      <th colspan="2">OCRBench</th>
      <th colspan="2">Overall</th>
    </tr>
    <tr>
      <th></th>
      <th>TEDS</th>
      <th>S-TEDS</th>
      <th>TEDS</th>
      <th>S-TEDS</th>
      <th>TEDS</th>
      <th>S-TEDS</th>
      <th>TEDS</th>
      <th>S-TEDS</th>
      <th>TEDS</th>
      <th>S-TEDS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="11">Expert TR models</td>
    </tr>
    <tr>
      <td>SLANNet-plus</td>
      <td>86.57</td>
      <td><b>96.43</b></td>
      <td>81.90</td>
      <td>89.08</td>
      <td>50.93</td>
      <td>65.84</td>
      <td>65.55</td>
      <td>77.73</td>
      <td>68.19</td>
      <td>79.21</td>
    </tr>
    <tr>
      <td>UniTable</td>
      <td>86.44</td>
      <td><u>95.66</u></td>
      <td>82.76</td>
      <td>89.82</td>
      <td>57.84</td>
      <td>70.47</td>
      <td>67.73</td>
      <td>78.65</td>
      <td>70.86</td>
      <td>80.81</td>
    </tr>
    <tr>
      <td colspan="11">General-purpose VLMs</td>
    </tr>
    <tr>
      <td>InternVL3.5-241B-A30B</td>
      <td>83.75</td>
      <td>88.76</td>
      <td>86.03</td>
      <td>90.53</td>
      <td>62.87</td>
      <td>69.52</td>
      <td>79.50</td>
      <td>85.81</td>
      <td>78.41</td>
      <td>84.18</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-72B</td>
      <td>84.39</td>
      <td>87.91</td>
      <td>87.85</td>
      <td>91.80</td>
      <td>81.22</td>
      <td>86.48</td>
      <td>81.33</td>
      <td>86.58</td>
      <td>83.52</td>
      <td>88.33</td>
    </tr>
    <tr>
      <td>Qwen3-VL-235B-A22B</td>
      <td>-</td>
      <td>-</td>
      <td>91.02</td>
      <td><u>94.97</u></td>
      <td>80.98</td>
      <td>86.19</td>
      <td>84.12</td>
      <td>88.15</td>
      <td>85.83</td>
      <td>90.07</td>
    </tr>
    <tr>
      <td>Gemini 2.5 Pro</td>
      <td>-</td>
      <td>-</td>
      <td>90.90</td>
      <td>94.32</td>
      <td><b>85.56</b></td>
      <td><u>90.07</u></td>
      <td>88.94</td>
      <td>89.47</td>
      <td><u>88.93</u></td>
      <td><u>91.23</u></td>
    </tr>
    <tr>
      <td>GPT-4o</td>
      <td>76.53</td>
      <td>86.16</td>
      <td>78.27</td>
      <td>84.56</td>
      <td>66.98</td>
      <td>79.04</td>
      <td>70.51</td>
      <td>79.55</td>
      <td>72.44</td>
      <td>81.15</td>
    </tr>
    <tr>
      <td>GPT-5</td>
      <td>-</td>
      <td>-</td>
      <td>84.91</td>
      <td>89.91</td>
      <td>63.25</td>
      <td>74.09</td>
      <td>79.91</td>
      <td>88.69</td>
      <td>78.30</td>
      <td>86.21</td>
    </tr>
    <tr>
      <td colspan="11">Document-parsing VLMs</td>
    </tr>
    <tr>
      <td>dots.ocr</td>
      <td>90.65</td>
      <td>93.76</td>
      <td>88.62</td>
      <td>92.86</td>
      <td>75.42</td>
      <td>81.65</td>
      <td>82.04</td>
      <td>86.27</td>
      <td>82.95</td>
      <td>87.58</td>
    </tr>
    <tr>
      <td>DeepSeek-OCR</td>
      <td>-</td>
      <td>-</td>
      <td>83.79</td>
      <td>87.86</td>
      <td>68.95</td>
      <td>75.22</td>
      <td>82.64</td>
      <td>87.33</td>
      <td>80.31</td>
      <td>85.11</td>
    </tr>
    <tr>
      <td>PaddleOCR-VL</td>
      <td>-</td>
      <td>-</td>
      <td><u>91.12</u></td>
      <td>94.62</td>
      <td>79.62</td>
      <td>85.04</td>
      <td>79.29</td>
      <td>83.93</td>
      <td>83.36</td>
      <td>87.77</td>
    </tr>
    <tr>
      <td>MinerU2.5</td>
      <td>89.07</td>
      <td>93.11</td>
      <td>90.85</td>
      <td>94.68</td>
      <td>79.76</td>
      <td>85.16</td>
      <td><u>87.13</u></td>
      <td><u>90.62</u></td>
      <td>86.82</td>
      <td>90.81</td>
    </tr>
    <tr>
      <td><b>TRivia-3B(Ours)</b></td>
      <td><b>91.79</b></td>
      <td>93.81</td>
      <td><b>91.60</b></td>
      <td><b>95.01</b></td>
      <td><u>84.90</u></td>
      <td><b>90.17</b></td>
      <td><b>90.76</b></td>
      <td><b>94.03</b></td>
      <td><b>89.88</b></td>
      <td><b>93.60</b></td>
    </tr>
  </tbody>
</table>
Overall一栏是三个基准上的加权平均分数：OmniDocBench v1.5, CC-OCR, and OCRBench v2.

# 环境配置
因为TRivia-3B是基于Qwen2.5-VL-3B进行训练，因此你可以参考[Qwen2.5-VL-3B installation guide](https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#quickstart) 进行环境配置。

我们强烈推荐安装[`vLLM >= 0.7.2`](https://github.com/vllm-project/vllm)来提高推理速度.

# 使用方法
TRivia-3B以表格图像作为输入并输出OTSL标记作为输出。

> 注意：TRivia-3B 是一个实验性的模型，没有经过严格的工程优化且无法输出LaTex公式或者以及表中有图片的场景。

## vLLM离线推理
确保已经安装 `vllm >= 0.7.2`. 将待识别的图片放到目录下并运行以下命令:

```bash
python run_vllm_offline_inf.py --ckpt_root opendatalab/TRivia-3B --image_root /path/to/images --output_path ./vllm_offline_output.json
# Examples
python run_vllm_offline_inf.py --ckpt_root opendatalab/TRivia-3B --image_root ./examples --output_path ./examples_output.json
```

输出是一个JSON文件([example](./example.json))，格式如下:
```json
[
    {
        "path": "...", // Image path
        "otsl": "...", // Unprocessed OTSL tags output by the model
        "html": "...", // Converted HTML tags
    }
]
```

## vLLM在线部署
你也可以使用vLLM或者SGLang部署TRivia-3B，并使用openai样式的api进行请求访问。

- 启动服务
```bash
vllm serve opendatalab/TRivia-3B --port 10000 --gpu_memory_utilization 0.8 
```
- Table Image Request
```python
import base64
from openai import OpenAI
from otsl_utils import convert_otsl_to_html

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:10000/v1",
    timeout=3600
)

image_path = "./examples/docstructbench_llm-raw-scihub-o.O-ijc.22994.pdf_3_5.png"
with open(path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "You are an AI specialized in recognizing and extracting table from images. Your mission is to analyze the table image and generate the result in OTSL format using specified tags. Output only the results without any other words and explanation." # Make sure to use this prompt for optimal performance.
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    }
]

response = client.chat.completions.create(
    model="opendatalab/TRivia-3B",
    messages=messages,
    temperature=0.0,
    max_tokens=8192
)
otsl_content = response.choices[0].message.content
html_content = convert_otsl_to_html(otsl_content)
print(f"Generated otsl tags: {otsl_content}")
print(f"HTML table: {html_content}")
```

## 

# Citation

```
@misc{zhang2025triviaselfsupervisedfinetuningvisionlanguage,
      title={TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition}, 
      author={Junyuan Zhang and Bin Wang and Qintong Zhang and Fan Wu and Zichen Wen and Jialin Lu and Junjie Shan and Ziqi Zhao and Shuya Yang and Ziling Wang and Ziyang Miao and Huaping Zhong and Yuhang Zang and Xiaoyi Dong and Ka-Ho Chow and Conghui He},
      year={2025},
      eprint={2512.01248},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.01248}, 
}
```


# License
[Apache License 2.0](LICENSE)

