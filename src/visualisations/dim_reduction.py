import os
import argparse
from typing import Type, Dict

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

import seaborn as sns
import plotly.express as px

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


THEME = "plotly_dark"
WIDTH = 1500
HEIGTH = 900
MY_PALETTE = ["#23aaff", "#ff6555", "#66c56c", "#f4b247"]
EN_CODES = {
    'Code Civil':'Civil Code',
    "Code d'Instruction Criminelle":'Code of Criminal Instruction',
    'Code Judiciaire':'Judicial Code',
    'La Constitution':'The Constitution',
    'Code de la Nationalité Belge':'Code of Belgian Nationality',
    'Code Pénal':'Penal Code',
    'Code Pénal Social':'Social Penal Code',
    'Code Pénal Militaire':'Military Penal Code',
    'Code de la Démocratie Locale et de la Décentralisation':'Code of Local Democracy and Decentralization',
    'Code de Droit Economique':'Code of Economic Law',
    'Codes des Droits et Taxes Divers':'Code of Rights and Taxes',
    'Code de Droit International Privé':'Code of Private International Law',
    'Code des Sociétés et des Associations':'Code of Companies and Associations',
    'Code du Bien-être au Travail':'Code of Workplace Welfare',
    'Code Electoral':'Electoral Code',
    'Code Consulaire':'Consular Code', 
    'Code Ferroviaire':'Railway Code',
    'Code de la Navigation':'Navigation Code',
    'Code Forestier':'Forestry Code',
    'Code Rural':'Rural Code',
    'Code de la Fonction Publique Wallonne':'Walloon Code of Public Service',
    "Code Wallon de l'Enseignement Fondamental et de l'Enseignement Secondaire":'Walloon Code of Basic and Secondary Education',
    "Code Wallon de l'Agriculture":"Walloon Code of Agriculture",
    "Code Wallon de l'Habitation Durable":"Walloon Code of Sustainable Housing",
    'Code Wallon du Bien-être des animaux':"Walloon Code of Animal Welfare",
    'Code Wallon du Développement Territorial':'Walloon Code of Territorial Development',
    "Code Wallon de l'Action sociale et de la Santé":"Walloon Code of Social Action and Health",
    "Code Réglementaire Wallon de l'Action sociale et de la Santé":"Walloon Regulatory Code of Social Action and Health",
    "Code Wallon de l'Environnement":"Walloon Code of Environment",
    "Code de l'Eau intégré au Code Wallon de l'Environnement":"Walloon Code of Water",
    "Code Bruxellois de l'Aménagement du Territoire":"Brussels Code of Spatial Planning",
    "Code Bruxellois de l'Air, du Climat et de la Maîtrise de l'Energie":'Brussels Code of Air, Climate and Energy',
    'Code Bruxellois du Logement':'Brussels Housing Code',
    'Code Electoral Communal Bruxellois':'Brussels Municipal Electoral Code'
}


def load_article_data(articles_path, vectors_path, codes=['The Constitution', 'Electoral Code', 'Railway Code', 'Forestry Code']):
    # Preprocess article dataframe.
    dfA = pd.read_csv(articles_path) #load article IDs and metadata.
    dfA = dfA[['id', 'code', 'reference']] #only keep code name and legal reference from metadata.
    dfA['code'] = dfA['code'].map(EN_CODES) #translate French code names to English.
    dfA = dfA.loc[dfA['code'].isin(codes)] #only keep articles from given codes.
    dfA['code_id'] = dfA.groupby(['code'], sort=False).ngroup() #give each code an ID.
    # dfA = dfA.sample(frac=0.05, replace=False, random_state=42)
    dfA.reset_index(drop=True, inplace=True) #reset index.

    # Load all node2vec vectors and extract the article vectors.
    n2v_vectors = KeyedVectors.load(vectors_path, mmap='r')
    art_vectors = []
    for i, row in dfA.iterrows():
        art_vectors.append(n2v_vectors[str(row['id'])])
    art_vectors = np.vstack(art_vectors)

    # Create new dataframe that merges article vectors and categories.
    cols = ['feat'+str(i+1) for i in range(art_vectors.shape[1])]
    dfE = pd.DataFrame(data=art_vectors[:,:], columns=cols)
    dfE = pd.concat([dfE, dfA], axis=1)
    return dfE


def decompose(vectors: Type[np.ndarray], config: Dict, n_comp: int=3, standardization: bool=True, seed: int=42):
    assert config['name'] in ['pca', 'tsne', 'umap'], f"Error: 'method' should be either 'pca', 'tsne', or 'umap'."
    if standardization:
        vectors = StandardScaler().fit_transform(vectors)
    if config['name'] == 'pca':
        reducer = PCA(n_components=n_comp, random_state=seed)
    elif config['name'] == 'tsne':
        vectors = PCA(n_components=50, random_state=seed).fit_transform(vectors)
        reducer = TSNE(n_components=n_comp, 
                       perplexity=config['perplexity'], 
                       learning_rate=config['lr'], 
                       n_iter=config['iter'], 
                       random_state=seed, verbose=0)
    elif config['name'] == 'umap':
        reducer = umap.UMAP(n_components=n_comp, n_neighbors=config['n_neighbors'], min_dist=config['min_dist'], metric=config['metric'], random_state=seed)
    return reducer.fit_transform(vectors)


def plot_reduction(df: Type[pd.DataFrame], vectors: Type[np.ndarray], config: Dict, output_dir: str, show: bool=False, save:bool=True):
    dfPC = df.copy(deep=True)
    dfPC['pc1'] = vectors[:,0]
    dfPC['pc2'] = vectors[:,1]
    dfPC['pc3'] = vectors[:,2]
    fig2d = (px.scatter(dfPC, 
                        x='pc1', y='pc2',
                        color='code',
                        custom_data=['reference'],
                        labels={'pc1':'x', 'pc2':'y', 'code': ''},
                        #title=f"{config['name']}",
                        color_discrete_sequence=MY_PALETTE,
                        width=WIDTH, height=HEIGTH, 
                        template=THEME)
            .update_traces(hovertemplate="<br>".join(["%{customdata[0]}"]))
            .update_layout(legend=dict(xanchor="right", x=0.99, yanchor="top", y=0.99))
    )
    fig3d = (px.scatter_3d(dfPC, 
                        x='pc1', y='pc2', z='pc3',
                        color='code', 
                        custom_data=['reference'],
                        labels={'pc1':'x', 'pc2':'y', 'pc3':'z', 'code': ''},
                        #title=f"{config['name']}",
                        color_discrete_sequence=MY_PALETTE,
                        width=WIDTH, height=HEIGTH, 
                        template=THEME)
            .update_traces(hovertemplate="<br>".join(["%{customdata[0]}"]))
            .update_layout(legend=dict(xanchor="right", x=0.99, yanchor="top", y=0.99))
    )
    if show:
        fig2d.show()
        fig3d.show()
    if save:
        if config['name'] == 'tsne':
            save_fig(fig2d, output_dir, f"{config['name']}_p{config['perplexity']}_2d.pdf")
            save_fig(fig3d, output_dir, f"{config['name']}_p{config['perplexity']}_3d.pdf")
        else:
            save_fig(fig2d, output_dir, f"{config['name']}_2d.pdf")
            save_fig(fig3d, output_dir, f"{config['name']}_3d.pdf")
    return


def save_fig(fig, output_dir, filename, width=WIDTH, height=HEIGTH):
    os.makedirs(output_dir, exist_ok=True)
    (fig.update_layout(template='simple_white',
                       width=width, height=height,
                       margin=dict(r=10, l=10, b=10, t=10),
                       legend=dict(x=0.9, y=0.99, bordercolor="#d3d3d3", borderwidth=0),
                       font=dict(family="Times New Roman, Times, serif", size=16, color="black"))
        .update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=False)
        .update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=False)
        .write_image(os.path.join(output_dir, filename))
    )


def main(args):
    # Load node vectors with categories.
    dataf = load_article_data(args.articles_path, args.vectors_path)

    # Configure different dimension reduction techniques.
    configs = [
        {'name': 'pca'}, 
        {'name': 'umap', 'n_neighbors': 15, 'min_dist': 0.1, 'metric':'euclidean'},
    ]
    for i in np.arange(5, 211, 5):
        configs.append({'name': 'tsne', 'perplexity': i, 'iter': 2000, 'lr': 200})

    # Run the techniques.
    for c in configs:
        out = decompose(vectors=dataf.iloc[:,:-5].values, config=c)
        plot_reduction(dataf, vectors=out, config=c, output_dir=args.out_dir, show=False, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles_path", 
                        type=str,
                        help="Path of the data file containing the law articles."
    )
    parser.add_argument("--vectors_path", 
                        type=str,
                        help="Path of the data file containing the article node2vec vectors."
    )
    parser.add_argument("--out_dir",
                        type=str,
                        help="Path of the output directory."
    )
    args, _ = parser.parse_known_args()
    main(args)
