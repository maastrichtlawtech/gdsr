import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Dict, Set


class BSARDataset(Dataset):
    def __init__(self, 
                 queries: pd.DataFrame, 
                 documents: pd.DataFrame,
                 stage: str, #train, dev, test
                 add_doc_title: Optional[bool] = False, #whether or not we should append the document title before its content.
                 hard_negatives: Optional[Dict[str, List[str]]] = None, #Dict[qid, List[neg_docid_i]].
                ):
        self.stage = stage
        self.add_doc_title = add_doc_title 
        self.hard_negatives = hard_negatives 
        self.queries = self.get_id_query_pairs(queries) #Dict[qid, query]
        self.documents = self.get_id_document_pairs(documents) #Dict[docid, document]
        self.one_to_one_pairs = self.get_one_to_one_relevant_pairs(queries) #List[(qid, pos_docid_i)]
        self.one_to_many_pairs =  self.get_one_to_many_relevant_pairs(queries) #Dict[qid, List[pos_docid_i]]

    def __len__(self):
        if self.stage == "train":
            return len(self.one_to_one_pairs)
        else:
            return len(self.one_to_many_pairs)

    def __getitem__(self, idx):
        pos_id, neg_id, pos_doc, neg_doc = (None, None, None, None)
        if self.stage == "train":
            # Get query and positive document.
            qid, pos_id = self.one_to_one_pairs[idx]
            query = self.queries[qid]
            pos_doc = self.documents[pos_id]
            if self.hard_negatives is not None:
                # Get one hard negative for the query (by poping the first one in the list and adding it back at the end).
                neg_id = self.hard_negatives[str(qid)].pop(0)
                neg_doc = self.documents[neg_id]
                self.hard_negatives[str(qid)].append(neg_id)
        else:
            qid, query = list(self.queries.items())[idx]
        return qid, pos_id, neg_id, query, pos_doc, neg_doc

    def get_id_query_pairs(self, queries: pd.DataFrame) -> Dict[str, str]:
        return queries.set_index('id')['question'].to_dict()

    def get_id_document_pairs(self, documents: pd.DataFrame) -> Dict[str, str]:
        if self.add_doc_title:
            documents['article'] = (documents['description'] + " [SEP] ").fillna('') + documents['article']
        return documents.set_index('id')['article'].to_dict()

    def get_one_to_one_relevant_pairs(self, queries: pd.DataFrame) -> List[Tuple[int, int]]:
        return (queries
                .assign(article_ids=lambda d: d['article_ids'].astype(str).str.split(','))
                .set_index(queries.columns.difference(['article_ids']).tolist())['article_ids']
                .apply(pd.Series)
                .stack()
                .reset_index()
                .drop(['category','subcategory','extra_description','question'], axis=1, errors='ignore')
                .filter(regex=r'^(?!level_).*$', axis=1)
                .rename(columns={0:'article_id','id':'question_id'})
                .apply(pd.to_numeric)
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
                .to_records(index=False))

    def get_one_to_many_relevant_pairs(self, queries: pd.DataFrame) -> Dict[str, List[str]]:
        return queries.set_index('id')['article_ids'].astype(str).str.split(',').apply(lambda x: [int(i) for i in x]).to_dict()
