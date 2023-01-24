import os
import re
import joblib
import argparse
from os.path import join, exists
from typing import Type, Dict, List, Optional
from tqdm import tqdm
tqdm.pandas(desc='Processing text')

import spacy
import numpy as np
import pandas as pd

import plotly.express as px

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TextPreprocessor():
    def __init__(self, spacy_model):
        self.nlp = spacy.load(spacy_model)

    def preprocess(self, series, lowercase=True, remove_punct=True, 
                   remove_num=True, remove_stop=True, lemmatize=True):
        return (series.progress_apply(lambda text: self.preprocess_text(text, lowercase, remove_punct, remove_num, remove_stop, lemmatize)))

    def preprocess_text(self, text, lowercase, remove_punct,
                        remove_num, remove_stop, lemmatize):
        if lowercase:
            text = self._lowercase(text)
        doc = self.nlp(text)
        if remove_punct:
            doc = self._remove_punctuation(doc)
        if remove_num:
            doc = self._remove_numbers(doc)
        if remove_stop:
            doc = self._remove_stop_words(doc)
        if lemmatize:
            text = self._lemmatize(doc)
        else:
            text = self._get_text(doc)
        return text

    def _lowercase(self, text):
        return text.lower()
    
    def _remove_punctuation(self, doc):
        return [t for t in doc if not t.is_punct]
    
    def _remove_numbers(self, doc):
        return [t for t in doc if not (t.is_digit or t.like_num or re.match('.*\d+', t.text))]

    def _remove_stop_words(self, doc):
        return [t for t in doc if not t.is_stop]

    def _lemmatize(self, doc):
        return ' '.join([t.lemma_ for t in doc])

    def _get_text(self, doc):
        return ' '.join([t.text for t in doc])



def compute_bow_embeddings(texts: List[str], method: str, output_dir: str, save: bool = True):
    # Load saved vectorizer if exists. Otherwise, create new vectorizer.
    if exists(join(output_dir, f'{method}_vectorizer.pkl')):
        vectorizer = joblib.load(join(output_dir, f'{method}_vectorizer.pkl'))
    else:
        if method == 'count':
            vectorizer = CountVectorizer(ngram_range=(1,1), max_features=10000, min_df=1, max_df=1.0)
        else:
            vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features=10000, min_df=1, max_df=1.0)
        vectorizer.fit(texts)

    # Vectorize texts.
    text_vectors = vectorizer.transform(texts)

    # Save vectorizer and vectors.
    if save:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(vectorizer, join(output_dir, f'{method}_vectorizer.pkl'))
        joblib.dump( text_vectors, join(output_dir, f'{method}_vectors.pkl'))
    return text_vectors


def plot_cosine_heatmap(vectors, save: bool = False, output_dir: str = None, title: str = None):
    """Plot cosine similarity matrix between the article vectors.
    """
    fig = (px.imshow(cosine_similarity(vectors, vectors),
                      labels={'color':''},
                      color_continuous_scale='bupu',#'burgyl',#'bupu',#'brwnyl',#'bluered',
                      #title=title,
                      width=825, height=800, template="plotly_dark")
              .update_traces(hovertemplate="<br><br>".join(["<b>Similarity</b>: %{z}", "<extra></extra>"]))
              .update_layout(yaxis_showticklabels=False, xaxis_showticklabels=False)
    )
    if save:
        save_fig(fig, output_dir, f'heatmap_{title}.pdf')
    return fig


def save_fig(fig, output_dir: str, filename: str, width: int = 880, height: int = 800):
    os.makedirs(output_dir, exist_ok=True)
    (fig.update_layout(template='simple_white',
                       width=width, height=height,
                       margin=dict(r=10, l=10, b=10, t=10),
                       legend=dict(x=0.9, y=0.99, bordercolor="#d3d3d3", borderwidth=0),
                       font=dict(family="Times New Roman, Times, serif", size=16, color="black"))
        .update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        .update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        .write_image(join(output_dir, filename))
    )


def main(args):
    # Load article dataframe.
    dfA = pd.read_csv(args.articles_path)[:100]

    # Clean articles.
    cleaner = TextPreprocessor(spacy_model="fr_core_news_md")
    dfA['processed_article'] = cleaner.preprocess(dfA['article'], lemmatize=True)

    # Vectorizing questions and articles with classic BoW.
    vectors = compute_bow_embeddings(
        texts=dfA['processed_article'].values.tolist(), 
        method=args.emb_method, 
        output_dir=args.out_dir
    )

    # Plot correlation.
    step = 200
    for i in np.arange(0, dfA.shape[0], step):
        plot_cosine_heatmap(vectors[i:i+step], save=True, output_dir=args.out_dir, title=f'{i}-{i+step}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles_path", 
                        type=str,
                        help="Path of the data file containing the law articles."
    )
    parser.add_argument("--out_dir",
                        type=str, 
                        help="Path of the output directory."
    )
    parser.add_argument("--emb_method",
                        type=str,
                        choices=["count", "tfidf"],
                        help="Method to compute vector representations."
    )
    args, _ = parser.parse_known_args()
    main(args)
