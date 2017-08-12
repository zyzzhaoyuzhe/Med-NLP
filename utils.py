import re
import pandas as pd
import numpy as np
from collections import OrderedDict, defaultdict
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from fuzzywuzzy import fuzz
import scipy.sparse
import itertools

# function definition
class Parser(object):
    def __init__(self, fuzzyName, sample_sent={}):
        self.df = pd.DataFrame()
        self.lastfound = []
        self.fuzzyName = fuzzyName
        self.sample_sent = sample_sent

    @staticmethod
    def preprocess(s):
        ## preprocess
        s = s.split('\n')
        s = map(string.lower, s)
        s = map(string.strip, s)
        # remove empty line
        s = list(filter(None, s))
        return s

    def parser(self, df_raw):
        for idx, s in df_raw['Report Text'].iteritems():
            # reset lastfound
            self.lastfound = ['study']
            s = self.preprocess(s)
            doc = self._parser(s)
            doc['Report Text'] = s
            self.df = self.df.append(doc, ignore_index=True)

    def _parser(self, s):
        doc = OrderedDict()
        # line number that contains
        for line in s:
            if ':' in line:
                name, content = line.split(':', 1)
                found = self.isvalidfield(name, )
                if found:
                    self.lastfound = found
                else:
                    content = line
            else:
                content = line
            if content:
                tostore = self.trueField(content)
                if tostore not in doc:
                    doc[tostore] = []
                doc[tostore].append(content)
        return doc

    def trueField(self, content):
        if set(self.lastfound) == set([u'findings', u'technique']):
            if fuzz.ratio(content, 'gray scale, color and pulsed doppler imaging were utilized.') > 70:
                return u'technique'
            else:
                return 'findings'
        else:
            return self.lastfound[0]

    def isvalidfield(self, name):
        found = []
        for k, v in self.fuzzyName.iteritems():
            if any(name in tmp or tmp in name for tmp in v):
                found.append(k)
        return found


def myisnumber(s):
    p = re.compile('^\d+(\.\d+)?$')
    return not p.match(s) is None


# def clean_velofield(s, prefix, velo_fields):
#     pre = ''
#     main = ''
#     for k, v in prefix.iteritems():
#         if k in s:
#             pre = v
#     for k, v in velo_fields.iteritems():
#         for vv in v:
#             if vv in s:
#                 main = k
#                 break
#     if main:
#         return '_'.join([pre, main])
#     else:
#         return ''

def find_field(s):
    velo_fields = {}
    velo_fields['cca'] = ['common carotid', 'cca']
    velo_fields['bulb'] = ['carotid bulb']
    velo_fields['eca'] = ['external carotid']
    velo_fields['ica'] = ['internal carotid', 'ica']
    for k, v in velo_fields:
        for vv in v:
            st = s.find(vv)
            if st != -1:
                ed = start + len(vv)
                return s[:start] + s[ed:], k
    return '', None


def find_dic(s, dic):
    output = []
    for k, v in dic.iteritems():
        for vv in v:
            st = s.find(vv)
            if st != -1:
                ed = st + len(vv)
                output.append((st, ed, k))
                break
    return output


def fieldnvelo(s):
    output = []
    velo_fields = {'cca': ['common carotid', 'cca'],
                   'bulb': ['carotid bulb'],
                   'eca': ['external carotid'],
                   'ica': ['internal carotid', 'ica']
                   }
    prefix = {'p': ['proximal', 'prox'],
              'd': ['distal', 'dist'],
              'm': ['middle', 'mid'],
              }
    re_fields = find_dic(s, velo_fields)
    re_prefix = find_dic(s, prefix)
    if re_fields:
        if re_prefix:
            for st, ed, k in re_prefix:
                m_dig = re.search(r'(\d+\.\d+)|\d+', s[ed:])
                m_occ = s[ed:].find('occluded')
                if m_occ != -1:
                    if m_dig and m_dig.start() < m_occ:
                        ve = float(m_dig.group())
                    else:
                        ve = 0
                else:
                    if m_dig:
                        ve = float(m_dig.group())
                    else:
                        ve = None
                if not ve is None:
                    output.append(('_'.join([k, re_fields[0][-1]]), ve))
        else:
            m_dig = re.search(r'(\d+\.\d+)|\d+', s[re_fields[0][1]:])
            m_occ = s[re_fields[0][1]:].find('occluded')
            if m_occ != -1:
                if m_dig:
                    if m_dig.start() < m_occ:
                        ve = float(m_dig.group())
                    else:
                        ve = 0
                else:
                    ve = 0
            else:
                if m_dig:
                    ve = float(m_dig.group())
                else:
                    ve = None
            if ve:
                output.append((re_fields[0][-1], ve))
    return output


# parse findings
def parse_findings(sents):
    if not isinstance(sents, list):
        return [], []
    flag_lr = ''
    velos = []
    text = []
    # new
    for idx, sent in enumerate(sents):
        tokens = word_tokenize(str(sent).translate(None, string.punctuation))
        # update right or left
        if 'left' in tokens and 'right' not in tokens:
            flag_lr = 'l'
        elif 'right' in tokens and 'left' not in tokens:
            flag_lr = 'r'
        # find field and velo
        fnv = fieldnvelo(sent)
        if fnv:
            if flag_lr:
                velos += [('_'.join([k, flag_lr]), v) for k, v in fnv]
            else:
                velos += fnv
        else:
            text.append(sent)
    return text, velos


# save to file
def save(df, filename):
    df_output = pd.DataFrame(df['Report Text'].tolist())
    df_output = df_output.applymap(lambda x: '\n'.join(x) if isinstance(x, list) else x)
    df_output.to_excel(filename)


# pos_tag to wordnet_pos
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def remove_puncdigit(s):
    s = str(s)
    s = s.translate(None, string.punctuation)
    s = s.translate(None, string.digits)
    return s

def list2str(l):
    s = ''
    for foo in l:
        s = ' '.join([s, remove_puncdigit(foo)])
    return s

#
def null2empty(df, field):
    # foo = df[(~df['Past'].isnull()) & (df['Past'] != 0)]
    if isinstance(field, basestring) and field in df:
        bar = df[field]
        bar.loc[bar.isnull()] = bar.loc[bar.isnull()].apply(lambda x: [])
        df[field] = bar
        return df
    elif isinstance(field, list):
        for f in field:
            if f not in df:
                continue
            bar = df[f]
            bar.loc[bar.isnull()] = bar.loc[bar.isnull()].apply(lambda x: [])
            df[f] = bar
    return df


## Get Bag of ngrams and transforms
import operator
from sklearn.feature_extraction.text import TfidfTransformer

def df2texts(df, field):
    texts = []
    for _, item in df.iterrows():
        if isinstance(item[field], basestring):
            foo = remove_puncdigit(item[field])
        elif isinstance(item[field], list):
            foo = ''
            for sent in item[field]:
                sent = remove_puncdigit(sent)
                foo = ' '.join([foo, sent])
        texts.append(foo.strip())
    return texts

def ngram_counter(texts, ngram=1, min_count=2):
    dic = defaultdict(int)
    for text in texts:
        words = text.split()
        for left in range(len(words)):
            for right in range(left+1, min(len(words)+1, left+ngram+1)):
                dic[' '.join(words[left:right])] += 1
    dic = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
    word2idx = {}
    idx2word = {}
    for idx, item in enumerate(dic):
        if item[1] < min_count:
            break
        word2idx[item[0]] = idx
        idx2word[idx] = item[0]
    return word2idx, idx2word

# text to matrix
def text2data(texts, word2idx, ngram):
    d = []  # counts
    r = []  # report index
    c = []  # word index
    for idx, text in enumerate(texts):
        foo = defaultdict(int)  # X: default value is 0
        words = text.split()
        for left in range(len(words)):
            for right in range(left+1, min(len(words)+1, left+ngram+1)):
                word = ' '.join(words[left:right])
                if word not in word2idx: continue
                foo[word2idx[word]] += 1
        for k, v in foo.iteritems():
            d.append(v)
            r.append(idx)
            c.append(k)
    data = scipy.sparse.csr_matrix((d, (r, c)), shape=(len(texts), len(word2idx)))
    return data

# matrix to text (bag of words)
def data2text(data, idx2word):
    cx = data.tocoo()
    text = [[] for _ in range(data.shape[0])]
    for i, j in itertools.izip(cx.row, cx.col):
        text[i].append(idx2word[j])
    return text

# raw frequency to tfidf
def data2tfidf(data):
    transform = TfidfTransformer()
    return transform.fit_transform(data)

# dataframe to tfidf wrapper (multi-fields)

class Df2TFIDF(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.ngram = None

    def fit(self, df, fields, ngram=1, min_count=2):
        self.ngram = ngram
        for field in fields:
            if field not in df.columns:
                continue
            texts = df2texts(df, field)
            word2idx, idx2word = ngram_counter(texts, ngram=ngram, min_count=min_count)
            self.word2idx[field] = word2idx
            self.idx2word[field] = idx2word

    def transform(self, df, fields):
        outputs = {}
        for field in fields:
            if field not in df.columns:
                continue
            texts = df2texts(df, field)
            word2idx, idx2word = self.word2idx[field], self.idx2word[field]
            bow_count = text2data(texts, word2idx, self.ngram)
            bow_tfidf = data2tfidf(bow_count)
            foo = {}
            foo['texts'] = texts
            foo['word2idx'] = word2idx
            foo['idx2word'] = idx2word
            foo['bow_count'] = bow_count
            foo['bow_tfidf'] = bow_tfidf
            outputs[field] = foo
        return outputs
