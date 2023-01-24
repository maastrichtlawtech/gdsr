import os, sys, csv
import argparse
from tqdm import tqdm
from typing import Union, List, Optional

import torch
import pandas as pd
from easynmt import EasyNMT
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class BackTranslator:
    def __init__(self, model_name: str):
        self.model = EasyNMT(model_name=model_name) #Best models to use: "m2m_100_1.2B", "opus-mt".
        
    def __call__(self, 
                 sentences: List[str],
                 sentence_ids: List[str],
                 sentence_labels: List[List[int]],
                 source_lang: str,
                 target_lang: str,
                 batch_size: int,
                 output_dir: str,
                 start_id: int,
                 show_progress: bool=True,
        ):
        # Create output file.
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'backtranslations_{source_lang}-{target_lang}.csv' ), 'a') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['id', 'question', 'article_ids']) #, 'source_id'])

        results = []
        new_id = start_id
        for batch_start in tqdm(
            iterable=range(0, len(sentences), batch_size),
            desc=f"- Back-translating batches of {batch_size} sentences", 
            disable=not show_progress, 
            leave=False,
        ):
            # Sample a batch of sentences.
            batch_sentences = sentences[batch_start:batch_start+batch_size]
            batch_sentence_ids = sentence_ids[batch_start:batch_start+batch_size]
            batch_sentence_labels = sentence_labels[batch_start:batch_start+batch_size]

            # Perform back-translation on batch.
            translations = self.model.translate(
                documents=batch_sentences, 
                source_lang=source_lang, 
                target_lang=target_lang,
            )
            back_translations = self.model.translate(
                documents=translations, 
                source_lang=target_lang, 
                target_lang=source_lang,
            )
            results.extend(back_translations)

            # Save batch output.
            with open(os.path.join(output_dir, f'paraphrases_{source_lang}-{target_lang}.csv' ), 'a') as outfile:
                writer = csv.writer(outfile)
                for new_s, s_labels, s_id in zip(back_translations, batch_sentence_labels, batch_sentence_ids):
                    writer.writerow([new_id, new_s, s_labels]) #, s_id])
                    new_id += 1
        return results


class QueryGenerator:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def __call__(self, 
                 documents: List[str],
                 document_ids: List[int],
                 decoding_strat: str,
                 max_length: int, 
                 num_results: int,
                 batch_size: int,
                 output_dir: str,
                 start_id: int,
                 show_progress: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ):
        # Set model to device.
        self.model.eval()
        self.model.to(device)

        # Create output file.
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'doc2query_samples.csv' ), 'a') as outfile:  
            writer = csv.writer(outfile)
            writer.writerow(['id', 'question', 'article_ids'])
        
        results = []
        new_id = start_id
        for batch_start in tqdm(
            iterable=range(0, len(documents), batch_size), 
            desc=f"- Generating queries for batches of {batch_size} documents", 
            disable=not show_progress, 
            leave=False,
        ):
            # Sample a batch of documents
            batch_documents = documents[batch_start:batch_start+batch_size]
            batch_document_ids = document_ids[batch_start:batch_start+batch_size]

            # Perform tokenization and load to device.
            inputs = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=batch_documents, 
                padding='longest', 
                truncation=True, 
                max_length=512, 
                return_tensors="pt",
            )
            input_ids = inputs['input_ids'].to(device)

            # Perform query generation.
            with torch.no_grad():
                if decoding_strat == 'greedy-search':
                    out = self.model.generate(input_ids=input_ids, max_length=max_length, num_return_sequences=num_results)
                elif decoding_strat == 'beam-search':
                    out = self.model.generate(
                        input_ids=input_ids, max_length=max_length, num_return_sequences=num_results,
                        num_beams=5, early_stopping=True, no_repeat_ngram_size=2,
                )
                elif decoding_strat == 'random-sampling':
                    out = self.model.generate(
                        input_ids=input_ids, max_length=max_length, num_return_sequences=num_results,
                        do_sample=True, top_k=0, temperature=0.7,
                    )
                elif decoding_strat == 'topk-sampling':
                    out = self.model.generate(
                        input_ids=input_ids, max_length=max_length, num_return_sequences=num_results,
                        do_sample=True, top_k=50,
                    )
                elif decoding_strat == 'nucleus-sampling':
                    out = self.model.generate(
                        input_ids=input_ids, max_length=max_length, num_return_sequences=num_results,
                        do_sample=True, top_k=0, top_p=0.95,
                    )
                elif decoding_strat == 'topk-nucleus-sampling':
                    out = self.model.generate(
                        input_ids=input_ids, max_length=max_length, num_return_sequences=num_results,
                        do_sample=True, top_k=50, top_p=0.95,
                    )
            gen_queries = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            gen_queries = [gen_queries[x:x+num_results] for x in range(0, len(gen_queries), num_results)]
            results.extend(gen_queries)

            # Save batch output.
            with open(os.path.join(output_dir, f'synthetic_queries.csv' ), 'a') as outfile:  
                writer = csv.writer(outfile)
                for queries, doc_id in zip(gen_queries, batch_document_ids):
                    for new_query in queries:
                        writer.writerow([new_id, new_query, doc_id])
                        new_id += 1
        return results


def main(args):
    df = pd.read_csv(args.texts_filepath)
    if args.augmentation == 'bt':
        paraphraser = BackTranslator('opus-mt')
        queries = paraphraser(
            sentences=df.question.values.tolist(),
            sentence_ids=df.id.values.tolist(),
            sentence_labels=df.article_ids.values.tolist(),
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            start_id=args.start_id,
        )
    elif args.augmentation == 'qg':
        generator = QueryGenerator('doc2query/msmarco-french-mt5-base-v1')
        queries = generator(
            documents=df.article.values.tolist(),
            document_ids=df.id.values.tolist(),
            decoding_strat=args.decoding_strat,
            max_length=args.max_query_length,
            num_results=args.num_gen_samples,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            start_id=args.start_id,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts_filepath", type=str, help="Path of the data file containing the texts.")
    parser.add_argument("--augmentation", type=str, choices=('bt', 'qg'), help="Type of data augmentation to perform: either back-translation ('bt') or query generation ('qg').")
    parser.add_argument("--output_dir",  type=str, help="Path of the output_directory.")
    parser.add_argument("--batch_size", type=int, help="Batch size for generation.")
    parser.add_argument("--start_id", type=int, help="The initial index to give the newly generated queries.")
    parser.add_argument("--source_lang", type=str, default=None, help="For Back-Translation, the language of the source queries.")
    parser.add_argument("--target_lang", type=str, default=None, help="For Back-Translation, the language to translate the queries to.")
    parser.add_argument("--decoding_strat", type=str, default=None, choices=('greedy-search', 'beam-search', 'random-sampling', 'topk-sampling', 'nucleus-sampling', 'topk-nucleus-sampling'), help="For Query Generation, the decoding strategy.")
    parser.add_argument("--max_query_length", type=int, default=None, help="For Query Generation, the maximum length of the newly generated queries.")
    parser.add_argument("--num_gen_samples", type=int, default=None, help="For Query Generation, the number of queries to generate per document.")
    args, _ = parser.parse_known_args()
    main(args)
