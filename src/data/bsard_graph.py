import re, os, sys
import argparse
from os.path import join, exists
from typing import Type, List, Optional, Dict

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

import torch
from torch import nn
import torch_geometric.data as geom_data

from src.common import utils
from src.data.long_tokenizer import LongTokenizer



class BSARDGraph(object):
    def __init__(self,
                 articles_filepath: str,
                 nodes_filepath: Optional[str] = None,
                 edges_filepath: Optional[str] = None,
                 output_dir: str = "output/graph",
        ):
        self.articles_filepath = articles_filepath
        self.nodes_filepath = nodes_filepath
        self.edges_filepath = edges_filepath
        self.output_dir = output_dir

    def build_pyg_graph(self,
                        tokenizer: Type[LongTokenizer], #text tokenizer.
                        encoder: Type[nn.Module], #text encoder.
                        artID_to_article: Dict[int, str], #mapping from article ID to article content (that might be prefixed by concatenation of headings).
        ):
        # Load articles and mapping from article ID to node ID.
        dfA = pd.read_csv(self.articles_filepath)
        artID_to_nodeID = dict(map(lambda t: (t[1], t[0]), enumerate(dfA['id'])))

        if (self.nodes_filepath is not None) and (self.edges_filepath is not None):
            # If legislation graph has already been created, load its nodes and edges.
            dfV = pd.read_csv(self.nodes_filepath, sep='|', header=None, names=["node"])
            dfE = pd.read_csv(self.edges_filepath, sep='|', header=None, names=["node1", "node2"])
        else:
            # Otherwise, create legislation graph from articles file.
            nodes, edges = self.create_graph(dfA, do_save=True)
            dfV = pd.DataFrame(nodes, columns=['node'])
            dfE = pd.DataFrame(edges, columns=['node1', 'node2'])

        # For nodes, replace article IDs by corresponding article contents.
        artID_to_article = {str(k): v for k, v in artID_to_article.items()}
        dfV.replace({'node': artID_to_article}, inplace=True)

        # For edges, replace "heading" nodes by corresponding node IDs.
        nodeID_to_node = dict(zip(dfV.index.to_series().apply(str), dfV.node))
        node_to_nodeID = {v: k for k, v in nodeID_to_node.items() if not v.isnumeric()}
        dfE.replace({'node1':node_to_nodeID, 'node2':node_to_nodeID}, inplace=True)

        # Create node feature matrix with shape [num_nodes, num_node_features], and graph connectivity in COO format with shape [2, num_edges].
        node_features = utils.encode(model=encoder, tokenizer=tokenizer, texts=dfV['node'], batch_size=512)
        edge_index = torch.tensor([dfE.node1.apply(int).tolist(), dfE.node2.apply(int).tolist()])
        G = geom_data.Data(x=node_features, edge_index=edge_index)
        return G, artID_to_nodeID

    def create_graph(self, df: Type[pd.DataFrame], do_save: bool):
        columns = ['law_type', 'code', 'book', 'part', 'act', 'chapter', 'section', 'subsection']
        df = self.clean(df, columns)
        nodes = self.create_nodes(df, ['id']+columns)
        edges = self.create_edges(df, columns+['id'])
        if do_save:
            self.save_nodes(nodes)
            self.save_edges(edges)
        return nodes, edges

    def plot_subgraph(self, source_node: str = "La constitution"):
        if not (self.nodes_filepath and self.edges_filepath):
            sys.exit("ERROR: Data files containing nodes and edges not found.")

        G = self.build_nx_graph(self.nodes_filepath, self.edges_filepath)
        net = Network(
            height='100%', width='100%', 
            bgcolor='#222222', font_color='white', 
            layout=True,
            heading=source_node,
        )
        net.set_options("""
            var options = {
                "layout": {
                    "hierarchical": {
                        "enabled": true,
                        "levelSeparation": 600,
                        "nodeSpacing": 100,
                        "treeSpacing": 410
                    }
                },
                "edges": {
                    "arrows": {
                        "to": {
                            "enabled": true
                        }
                    }
                },
                "interaction": {
                    "hover": true
                },
                "physics": {
                    "enabled": false,
                    "hierarchicalRepulsion": {
                        "centralGravity": 0
                    },
                    "minVelocity": 0.75,
                    "solver": "hierarchicalRepulsion"
                }
            }
        """)
        #net.show_buttons(filter_=True)
        g = G.subgraph(nx.descendants(G, source_node))
        net.from_nx(g)
        os.makedirs(output_path, exist_ok=True)
        net.show(join(output_path, source_node.replace(' ', '_').lower() + '.html'))

    def clean(self, df:Type[pd.DataFrame], columns: List[str]):
        for c in columns:
            df[c] = df[c].apply(lambda x: self.__clean_heading(x) if not pd.isna(x) else x)
        return df

    def __clean_heading(self, x: str):
        x = re.sub(r'\(note\s*:.*\)', '', str(x))
        x = re.sub(r'[\(<](inséré|l).*[>\)]', '', x)
        x = re.sub(r'\d+.*;\s*en\s*vigueur\s*\:.*\d+>?', '', x)
        x = re.sub(r'[\.]$', '', x)
        x = re.sub(r'^[-{:]', '', x)
        x = re.sub(r'[\(\)\"]', '', x)
        x = re.sub(r'\.\s+-', ' -', x)
        x = re.sub(r'\.\s*', ', ', x)
        x = re.sub(r'_', ' ', x)
        x = x.strip().capitalize()
        return x if x else float('nan')

    def create_nodes(self, df: Type[pd.DataFrame], columns: List[str]):
        nodes = []
        for c in columns:
            nodes.extend(df[c].dropna().unique().tolist())
        nodes = list(dict.fromkeys(nodes))
        return nodes

    def create_edges(self, df: Type[pd.DataFrame], columns: List[str]):
        chained_lists = df.loc[:, columns].values.tolist()
        chained_lists = [[x for x in lst if str(x) != 'nan'] for lst in chained_lists]
        edges = [(x, y) for lst in chained_lists for x, y in zip(lst, lst[1:])]
        edges = list(dict.fromkeys(edges))
        return edges

    def save_nodes(self, nodes):
        with open(join(self.output_dir, 'nodes.txt'), 'w') as f:
            for n in nodes:
                f.write(f'{n}\n')

    def save_edges(self, edges):
        with open(join(self.output_dir, 'edges.txt'), 'w') as f:
            for n1, n2 in edges:
                f.write(f'{n1}|{n2}\n')

    def build_nx_graph(self, nodes_filepath: str, edges_filepath: str):
        G = nx.DiGraph()
        nodes = nx.read_adjlist(nodes_filepath, delimiter='|')
        edges = nx.read_edgelist(edges_filepath, delimiter='|')
        G.add_nodes_from(nodes)
        G.add_edges_from(edges.edges())
        return G
