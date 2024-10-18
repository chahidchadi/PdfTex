from transformers import AutoProcessor, VisionEncoderDecoderModel
from pathlib import Path
import re


def final_code_generator(latex_output):
    code1 = r"""\documentclass[12pt,a4paper]{article}
% Language and font encoding
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}

% AMS packages for math
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}

% Geometry for margins
\usepackage{geometry}
\geometry{a4paper, margin=1in}

% Package for including graphics
\usepackage{graphicx}

% Package for tables
\usepackage{array}

% Package for colors
\usepackage{xcolor}

% Package for hyperlinks
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    pdftitle={Math Document},
    bookmarks=true,
    pdfpagemode=FullScreen,
}

% Package for including code
\usepackage{listings}
\lstset{
    basicstyle=\ttfamily,
    columns=fullflexible,
    breaklines=true
}

% Theorem environments
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{proposition}[theorem]{Proposition}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{example}[theorem]{Example}

% Custom commands
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}

\begin{document}
"""
    return f"{code1}\n{latex_output}\n\\end{{document}}"


def models():
 # Define local paths for saving the model and processor
 local_model_path = Path("./local_nougat_model")
 local_processor_path = Path("./local_nougat_processor")
# Download and save the model
 print("Downloading and saving the model...")
 model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")
 model.save_pretrained(local_model_path)
# Download and save the processor
 print("Downloading and saving the processor...")
 processor = AutoProcessor.from_pretrained("facebook/nougat-small")
 processor.save_pretrained(local_processor_path)
 print(f"Model saved to: {local_model_path}")
 print(f"Processor saved to: {local_processor_path}")
 print("Download complete.")
 return model , processor

def replace_latex_notation(text):
    """
    Replace LaTeX-style math notation in the given text with Markdown-compatible syntax.

    Parameters:
    text (str): The input text containing LaTeX-style math notation.

    Returns:
    str: The modified text with LaTeX-style math notation replaced.
    """
    text = text.replace(r"\(", "$")
    text = text.replace(r"\)", "$")
    text = text.replace(r"\[", "$$")
    text = text.replace(r"\]", "$$")
    return text
