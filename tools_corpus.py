import dhlab as dh
import dhlab.api.dhlab_api as api
import pandas as pd
import dhlab.graph_networkx_louvain as gnl
import networkx as nx
import requests
import json
from io import StringIO

def imag_corpus():
    res =  requests.get(f"{dh.constants.BASE_URL}/imagination/all")
    if res.status_code == 200:
        data = json.loads(res.text)
    else:
        data = "[]"
    return pd.DataFrame(data)

def get_imag_corpus():
    im = imag_corpus()
    c = dh.Corpus()
    c.extend_from_identifiers(im.urn)
    corpus = c.frame
    corpus.dhlabid = corpus.dhlabid.astype(int)
    corpus = corpus[['urn', 'dhlabid', 'title', 'authors', 
       'city',  'year', 'publisher', 'langs', 'subjects', 'ddc',
       'genres', 'literaryform', 'doctype', 'ocr_creator']]
    corpus = corpus.merge(im[['urn', 'category']], left_on='urn', right_on='urn')
    return corpus

def geo_locations(dhlabid):
    res = requests.get(f"{dh.constants.BASE_URL}/imagination_geo_data", params={"dhlabid":dhlabid})
    if res.status_code == 200:
        data = pd.read_json(StringIO(res.text))
    else:
        data = pd.DataFrame()
    return data

def make_collocation_graph(corpus, target_word, top=15, before=4, after=4, ref = None, limit=1000):
    """Make a cascaded network of collocations ref is a frequency list"""
    

    coll = dh.Collocations(corpus, [target_word], before=before, after=after, samplesize=limit).frame
    
    if not ref is None:
        coll['counts'] = coll['counts']*100/coll['counts'].sum()
        coll['counts'] = coll['counts']/ref
        
    coll = coll.sort_values(by="counts", ascending = False)
    edges = []
    visited = []               
    for word in coll[:top].index:
        # loop gjennom kollokasjonen og lag en ny kollokasjon for hvert ord
        edges.append((target_word, word, coll.loc[word]))
        if word.isalpha() and not word in visited:
            subcoll = dh.Collocations(corpus, [word], before=before, after=after, samplesize=limit).frame        
            if not ref is None:
                subcoll['counts'] = subcoll['counts']*100/subcoll['counts'].sum()
                subcoll['counts'] = subcoll['counts']/ref

            for w in subcoll.sort_values(by="counts", ascending = False)[:top].index:
                if w.isalpha():
                    edges.append((word, w, subcoll.loc[w]))
            visited.append(word)
            #print(visited)
    target_graph = nx.Graph()
    target_graph.add_edges_from(edges)

    return target_graph

def make_imagination_corpus():
    """Bygg hele imagination-korpuset fra dhlab"""
    return get_imag_corpus()

def imagination_ngram(corpus, words, mode='rel'):
    return corpus_ngram(corpus, words, mode)
    
def corpus_ngram_old(corpus, words, mode='rel'):
    cnts = api.get_document_frequencies(list(corpus.urn), words = words)
    d2y = pd.Series(corpus.set_index('dhlabid')['year'].to_dict())
    d2y.to_frame('year')
    if mode.startswith('r') or mode.startswith('R'):
        df = cnts['relfreq']
        frek = df.transpose().copy()
        frek = pd.concat([frek, d2y.to_frame('year')], axis = 1).groupby('year').mean()
    else:
        df = cnts['freq']
        frek = df.transpose().copy()
        frek = pd.concat([frek, d2y.to_frame('year')], axis = 1).groupby('year').sum()
        frek = frek.astype(int)
        
    frek.index = frek.index.astype(int)
    return frek

def corpus_ngram(
        corpus: pd.DataFrame,
        words: str,
        mode: str = 'rel'
):
    """Extract corpus-specific frequencies for given n-grams"""
    # Split the input words into a list of words
    search_terms = words.split(" ")

    # Create dataframe where the relevant years from the corpus are the index
    d2y = pd.Series(corpus.set_index('dhlabid')['year'].to_dict()).to_frame("year")
    # print(d2y)

    # Fetch frequencies from the corpus documents
    counts = api.get_document_frequencies(list(corpus.urn), words=search_terms)

    absfreq = counts['freq'].transpose().copy()
    absfreq_by_year = pd.concat([d2y, absfreq], axis=1
        ).groupby('year').sum().convert_dtypes()

    if mode.lower().startswith('r'):
        relfreq = counts['relfreq'].transpose().copy()
        # Calculate the total word frequency per urn
        urncounts = (absfreq / relfreq)
        # Group counts by year
        urncounts_by_year = pd.concat([d2y, urncounts], axis=1).groupby('year').sum()
        # Calculate relative frequency per year
        frek = (absfreq_by_year / urncounts_by_year) * 100
        # Ensure NaN-values are set to 0
        frek = frek.astype(float).fillna(0.0).astype(float)
    else:
        frek = absfreq_by_year.fillna(0)
    frek.index = frek.index.astype(int)
    return frek
    
class URN_Ngram():
    def __init__(self, corpus=None, wordbags=None):
        """corpus must be a pandas dataframe with at least a urn column a dhlabid column 
        and column for aggregation, e.g. """
        self.corpus = corpus
        self.wordbags = wordbags
        self.counts = api.evaluate_documents(wordbags=self.wordbags, urns= corpus.urn.to_list()).fillna(0)
        self.urn_freq = api.get_urn_frequencies(urns=corpus.urn.to_list()).set_index('urn')
        
        # convert index to ints
        self.urn_freq.index = self.urn_freq.index.astype(int)
        self.counts.index = self.counts.index.astype(int)
        
        self.agg = dict()
        self.percent = dict()
        
        self.dataframe = self.counts.merge(self.urn_freq, left_index=True, right_index=True)
        
    def aggregate(self, agg_column=None):
        """Aggregate over column"""
        if agg_column not in self.agg:
            agg = self.corpus[['dhlabid',agg_column]].set_index('dhlabid').merge(self.dataframe, left_index=True, right_index=True)
            #self.agg['agg_column'] 
            summing = agg.groupby(agg_column).sum()
            self.agg[agg_column] = summing
        return self.agg[agg_column]

    def percents(self, agg_column=None):
        if agg_column not in self.agg:
            self.aggregate(agg_column)
        if agg_column not in self.percent:
            df = self.aggregate(agg_column)
            df.loc[:, df.columns != 'freq'] = df.div(df['freq'], axis=0).loc[:, df.columns != 'freq']
            self.percent[agg_column] = df.loc[:, df.columns != 'freq']*100
        return self.percent[agg_column]
        
def get_text(urn):
    res = requests.get(f"https://api.nb.no/dhlab/freetext/get_text?urn={urn}")
    if res.status_code == 200:
        result = res.content.decode('unicode_escape')
    else:
        result = ""
    return result