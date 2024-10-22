from platform import win32_edition
import numpy as np
import cupy as cp
import scipy
import os
from os.path import exists, isfile, join
from pathlib import Path
import sys
import shutil
import gc
import math
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import json
from math import floor
np.set_printoptions(precision=9)

# Import stopwords
import nltk
from nltk import word_tokenize
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.util import everygrams

# Import TensorLy
import tensorly as tl
import cudf
from cudf import Series
from cuml.feature_extraction.text import CountVectorizer # replace cuml with sklearn equivalents 
#from cuml.preprocessing.text.stem import PorterStemmer
from nltk.stem import PorterStemmer
import cupyx 

#Insert Plotly
import pandas as pd
import modin.pandas as md
import time
import pickle






# Import utility functions from other files
import file_operations as fop



# Constants

ROOT_DIR        = "/home/debanks/Dropbox/CongSpeechData/hein-bound/" # raw data
INDIR           = "processed_speech/"
RAW_DATA_PREFIX = "processed_speech/" # save the processed data


# Output Relative paths -- do not change
X_MAT_FILEPATH_PREFIX = "x_mat/"
X_FILEPATH = "X_full.obj"
X_DF_FILEPATH = "X_df.obj"
X_LST_FILEPATH = "X_lst.obj"
CORPUS_FILEPATH_PREFIX = "corpus/"
GENSIM_CORPUS_FILEPATH = "corpus.obj"
COUNTVECTOR_FILEPATH = "countvec.obj"
TOP_SENTS_FILEPATH = "top_sents.obj"
JST_FILEPATH = "JST.obj"
VOCAB_FILEPATH = "vocab.csv"
EXISTING_VOCAB_FILEPATH = "vocab.obj"
TOPIC_FILEPATH_PREFIX   = 'predicted_topics/'
DOCUMENT_TOPIC_FILEPATH = 'dtm.csv'
COHERENCE_FILEPATH = 'coherence.obj'
DOCUMENT_TOPIC_FILEPATH_TOT = 'dtm_df.csv'
OUT_ID_DATA_PREFIX = 'ids/' 
TOP_WORDS_FILEPATH ='top_words.csv'

# Device settings
backend="cupy"
tl.set_backend(backend)
device = 'cuda'
porter_stemmer = PorterStemmer()


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)




def partial_fit(self , data):
    if(hasattr(self , 'vocabulary_')):
        vocab = self.vocabulary_ # series
    else:
        vocab = Series()
    self.fit(data)
    vocab = vocab.append(self.vocabulary_)
    self.vocabulary_ = vocab.unique()

def tune_filesplit_size_on_IPCA_batch_size(IPCA_batchsize):
    return None



RAW_DATA_PREFIX = "processed_speech/"

# Device settings
backend="cupy"
tl.set_backend(backend)
device = 'cuda'
porter_stemmer = PorterStemmer()

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def basic_clean(df):
    df['speech'] = df['speech'].astype('str')
    df = df.drop_duplicates(keep="first")
    df['speech'] = df['speech'].str.lower()
    df['speech'] = df['speech'].str.replace(r'[^\w\s\d]+', '')
    return df


inDir = os.path.join(ROOT_DIR, INDIR)

dl = sorted(fop.get_files_in_dir(inDir)) # we sort so that they are ordered in chronological order
print("Done. Split files located at: {}.\n".format(inDir))


for i, f in enumerate(dl):
    path_in      = os.path.join(inDir,f)
    print(path_in)
    df = md.read_csv(path_in, sep="|",encoding='cp1252',on_bad_lines='skip' ,engine="python" ,names = ['speech_id','speech'],usecols=['speech_id','speech'])
    df = df[df['speech'].notnull()] # remove empty speeches
    df['speech'] = df['speech'].apply(stem_sentences) # stem the speeches
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    df   = basic_clean(df) # remove duplicates and punctuation
    mask = df['speech'].str.len() > 15 #more than 15 characters
    df   = df.loc[mask]
    print(df.head())
    df.to_csv(os.path.join(ROOT_DIR, RAW_DATA_PREFIX + f), index=False) 
    
for i, f in enumerate(dl):
    source_path = os.path.join(ROOT_DIR, RAW_DATA_PREFIX + f)
    for j,chunk in enumerate(pd.read_csv(source_path, chunksize=200000)):
        nrows = chunk.shape[0]
        if nrows==200000:
            chunk.to_csv(os.path.join(ROOT_DIR, RAW_DATA_PREFIX + Path(f).stem+'_'+str(j)+'.txt'), index=False) 
        if j>0:
            chunk.to_csv(os.path.join(ROOT_DIR, RAW_DATA_PREFIX + Path(f).stem+'_'+str(j)+'.txt'), index=False) 



# declare the stop words 

stop_words  = (stopwords.words('english'))
added_words = ["web3", "dominate", "dominating", "premnt", "youtube.com", "youtu.be", "uniswap hack", "uniswapexploit",
               "revoke cash", "giveaway", "sushiswap", "TheHyperVerse.net", "TheHyperVerse", "YieldNodes.com", "COTPS.com", "GoArbit.com",
               "JuicyFields.io", "Unique-Exchange.co","TriumphFX.com", "CryptoCrown.vip", "freeway.io", "CashFXGroup.com", "Wintermute", "Elon Musk"]

# set stop words and countvectorizer method
stop_words= list(np.append(stop_words,added_words))
CountVectorizer.partial_fit = partial_fit



def custom_preprocessor(doc):
    return doc

countvec = CountVectorizer( stop_words = stop_words, #stop_words, # works
                            lowercase = True,#True, # works
                            ngram_range = (1, 2), #(1,2), ## allow for bigrams
                            preprocessor = custom_preprocessor,
                            max_df = 500000, #100000, # limit this to 10,000 ## 500000 for 8M
                            min_df = 20)# 2000) ## limit this to 20 ## 2500 for 8M