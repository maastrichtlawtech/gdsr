import os
import timeit, datetime
from pathlib import Path
from functools import wraps
from os.path import join, dirname
from argparse import ArgumentTypeError
from typing import List, Type, Union, Optional

import csv
import pandas as pd

from src.pl_modules.biencoder_module import BiencoderModule


def log_step(funct):
    @wraps(funct)
    def wrapper(*args, **kwargs):
        tic = timeit.default_timer()
        result = funct(*args, **kwargs)
        time_taken = datetime.timedelta(seconds=timeit.default_timer() - tic)
        print(f"Just ran '{funct.__name__}' function. Took: {time_taken}")
        return result
    return wrapper

def save_with_huggingface(ckpt_path: str):
    """Save checkpoint with Hugging Face saving function.
    """
    biencoder = BiencoderModule.load_from_checkpoint(ckpt_path)
    biencoder.q_encoder.word_encoder.save_pretrained(join(dirname(ckpt_path), Path(ckpt_path).stem + "__hf", "q_encoder"))
    biencoder.q_tokenizer.tokenizer.save_pretrained(join(dirname(ckpt_path), Path(ckpt_path).stem + "__hf", "q_encoder"))
    biencoder.d_encoder.word_encoder.save_pretrained(join(dirname(ckpt_path), Path(ckpt_path).stem + "__hf", "d_encoder"))
    biencoder.d_tokenizer.tokenizer.save_pretrained(join(dirname(ckpt_path), Path(ckpt_path).stem + "__hf", "d_encoder"))

def corpus2txt(filepath: str, out_dir: str):
    """Extract articles from BSARD and save them in same .txt file (one article per line).
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(filepath)[['article']]
    df.to_csv(
        join(out_dir, "articles_fr.txt"),
        index=False, header=False,
        sep="|", escapechar='\\',
        quoting=csv.QUOTE_NONE
    )

def str2bool(v: str):
    """From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
