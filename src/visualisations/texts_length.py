import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns

from transformers import AutoTokenizer


def draw_distributions(dfA, dfQ, stat_type: str, out_dir: str):
    # Data.
    df = pd.DataFrame(dict(
        num_tokens=np.concatenate((dfQ['num_tokens'], dfA['num_tokens'])),
        text_type=np.concatenate((["Question"]*len(dfQ['num_tokens']), ["Article"]*len(dfA['num_tokens'])))
    ))

    # Plot.
    fig = sns.histplot(data=df, x="num_tokens", hue="text_type", log_scale=True, stat=stat_type, common_norm=False)

    # Labels.
    fig.set(xlabel="Number of tokens (log scale)", 
            ylabel=f"{stat_type.capitalize()} of samples" if stat_type != 'count' else "Number of samples",
            xlim=(2,10e3), 
            ylim=(0.0, 0.16) if stat_type == 'proportion' else ((0.0, 1000) if stat_type == 'count' else (0.0, None))
    )
    
    # Legend.
    fig.legend_.set_title(None)
    sns.move_legend(fig, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.12), frameon=False)

    # Style.
    sns.despine(offset={'bottom':5, 'left':5})
    sns.set(font="Times New Roman", font_scale=1., style="ticks", rc={'figure.figsize':(6,5)})

    # Saving.
    os.makedirs(out_dir, exist_ok=True)
    fig.get_figure().savefig(os.path.join(out_dir, f"length_distributions_{stat_type}.pdf"), bbox_inches='tight')


def extract_tokens(tokenizer, dataf, text_column: str):
    dataf['tokens'] = tokenizer(dataf[text_column].tolist(), padding=False, truncation=False)['input_ids']
    dataf['num_tokens'] = dataf['tokens'].str.len()
    return dataf


def main(args):
    # Load questions and articles.
    dfA = pd.read_csv(args.articles_path)
    dfQ = pd.concat([
        pd.read_csv(args.train_questions_path), 
        pd.read_csv(args.test_questions_path)
    ])

    # Load tokenizer, tokenize questions and articles, and compute number of tokens per text.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    dfQ = extract_tokens(tokenizer, dfQ, 'question')
    dfA = extract_tokens(tokenizer, dfA, 'article')

    # Draw token length distribution.
    draw_distributions(dfA, dfQ, stat_type=args.stat_type, out_dir=args.out_dir)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles_path", 
                        type=str, 
                        help="Path of the data file containing the law articles."
    )
    parser.add_argument("--train_questions_path", 
                        type=str, 
                        help="Path of the data file containing the training questions."
    )
    parser.add_argument("--test_questions_path", 
                        type=str, 
                        help="Path of the data file containing the test questions."
    )
    parser.add_argument("--tokenizer_name", 
                        type=str,
                        help="Name of the HF tokenizer."
    )
    parser.add_argument("--stat_type", 
                        type=str,
                        choices=['count','frequency','proportion','percent','density'],
                        help="Aggregate statistic to compute in each bin."
    )
    parser.add_argument("--out_dir",
                        type=str,
                        help="Path of the output directory."
    )
    args, _ = parser.parse_known_args()
    main(args)
