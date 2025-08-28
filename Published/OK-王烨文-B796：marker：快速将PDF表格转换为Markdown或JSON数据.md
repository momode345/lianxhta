# B796：marker：快速将PDF表格转换为Markdown或JSON数据

- [github](https://github.com/vikParuchuri/marker)

Marker converts documents to markdown, JSON, and HTML quickly and accurately.

-   Converts PDF, image, PPTX, DOCX, XLSX, HTML, EPUB files in all languages
-   Does structured extraction, given a JSON schema (beta)
-   Formats tables, forms, equations, inline math, links, references, 和 code blocks
-   Extracts and saves images
-   Removes headers/footers/other artifacts
-   Extensible with your own formatting and logic
-   Optionally boost accuracy with LLMs
-   Works on GPU, CPU, or MPS

## Performance


![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250530101756.png)

Marker benchmarks favorably compared to cloud services like Llamaparse and Mathpix, as well as other open source tools.

The above results are running single PDF pages serially. Marker is significantly faster when running in batch mode, with a projected throughput of 122 pages/second on an H100 (.18 seconds per page across 22 processes).

As you can see, the use\_llm mode offers higher accuracy than marker or gemini alone.

## Examples


| PDF | File type | Markdown | JSON |
| --- |  --- |  --- |  --- |
| [Think Python](https://greenteapress.com/thinkpython/thinkpython.pdf) | Textbook | [View](https://github.com/VikParuchuri/marker/blob/master/data/examples/markdown/thinkpython/thinkpython.md) | [View](https://github.com/VikParuchuri/marker/blob/master/data/examples/json/thinkpython.json) |
| [Switch Transformers](https://arxiv.org/pdf/2101.03961.pdf) | arXiv paper | [View](https://github.com/VikParuchuri/marker/blob/master/data/examples/markdown/switch_transformers/switch_trans.md) | [View](https://github.com/VikParuchuri/marker/blob/master/data/examples/json/switch_trans.json) |
| [Multi-column CNN](https://arxiv.org/pdf/1804.07821.pdf) | arXiv paper | [View](https://github.com/VikParuchuri/marker/blob/master/data/examples/markdown/multicolcnn/multicolcnn.md) | [View](https://github.com/VikParuchuri/marker/blob/master/data/examples/json/multicolcnn.json) |

## Installation

You'll need python 3.10+ and PyTorch. You may need to install the CPU version of torch first if you're not using a Mac or a GPU machine. See [here](https://pytorch.org/get-started/locally/) for more details.

![](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/20250530173405.png)

复制上图生成的代码，通过命令行运行：

- 按快捷键 Win+R，输入 `cmd`，打开命令行窗口
- 贴入 `pip3 install torch torchvision torchaudio` 命令。

然后，使用如下命令安装 `marker` 包:

```
pip install marker-pdf
```

If you want to use marker on documents other than PDFs, you will need to install additional dependencies with:

```
pip install marker-pdf[full]
```
