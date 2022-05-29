from typing import Union
import pandas as pd
import swifter
import numpy as np
from PIL import Image
import argparse
import os
import string

parser = argparse.ArgumentParser('Tool for merging the two datasets')
parser.add_argument(
    '--sarcastic-path', 
    type=str, 
    default='../ROData/sarcastic_articles_anonymized_step2.csv',
    help='The path to the dataset from Times New Roman'
)
parser.add_argument(
    '--non-sarcastic-path', 
    type=str, 
    default='../ROData/non-sarcastic_articles_anonymized_step2.csv',
    help='The path to the dataset from Stiri Pe Surse'
)
parser.add_argument(
    '--output-path', 
    type=str, 
    default='../ROData/merged.csv',
    help='Where to save the merged dataset'
)
parser.add_argument(
    '--use-title',
    action='store_true',
    help='Use only the title of the article as the text component'
)
args = parser.parse_args()

translation_dict = {
    'economie': 'economy',
    'externe': 'global_news',
    'politica': 'politics',
    'politic': 'politics',
    'sanatate': 'health',
    'social': 'social',
    'life-death': 'social',
    'monden': 'social',
    'sport': 'sports',
    'uniunea-europeana': 'eu_news',
    'it-stiinta': 'science',
}

def crop_path(image_path: str):
    tokens = image_path.split('/')[3:]
    return '/'.join(tokens)

def get_id(image_path: str):
    return image_path.split('/')[2].split('.')[0]

def translate_topic(topic: str):
    return translation_dict[topic]

def remove_whitespace(text: str):
    return (' '.join(text.split())).replace('\n', '')

def return_first_n_sentences(text: str, n = 1):
    if n < 1:
        raise ValueError('Number of sentences must be >1')

    sentence_terminators = set(['.', '!', '?', ';', ':'])
    
    n_found = 0
    for i in range(len(text)):
        if text[i] in sentence_terminators:
            n_found += 1
            if n_found == n:
                return text[:i+1]

    return text

def return_first_n_words(text: str, n = 20):
    if n < 1:
        raise ValueError('Number of words must be >1')

    words = text.split(' ')

    i = 0
    while i < len(words):
        if len(words[i]) == 0:
            words.pop(i)
        else:
            i += 1

    if len(words) > n:
        words = words[:n]

    return ' '.join(words)

def process_text(text: str):
    text = text.lower()
    # text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def verify_text(text: str):
    if not isinstance(text, str):
        return False
    if len(text) > 3000:
        return False
    if len(text) < 40:
        return False
    return True

def verify_topics(topic: str):
    if topic in translation_dict:
        return True
    return False

def verify_images(image_path: str, images_dir_path = '../ROData/images'):
    _path = os.path.abspath(os.path.join(images_dir_path, image_path))
    
    try:
        Image.open(_path).convert('RGB')
    except:
        return False

    return True

if args.use_title:
    text_source = 'title_anonymized'
else:
    text_source = 'article_anonymized'

df_sarcasm = pd.read_csv(args.sarcastic_path, low_memory=False)
df_nonsarc = pd.read_csv(args.non_sarcastic_path, low_memory=False)
df_sarcasm['sarcastic'] = 'yes'
df_nonsarc['sarcastic'] = 'no'

# select the columns to be kept from both .csv files
df_sarcasm = df_sarcasm[
    ['sarcastic', 'topic', 'photo_path', 'url', text_source]
]
df_nonsarc = df_nonsarc[
    ['sarcastic', 'topic', 'photo_path', 'url', text_source]
]

# rename the columns for compatibility
df_sarcasm = df_sarcasm.rename(
    columns={
        text_source: 'text',
        'photo_path': 'image_path'
    }
)
df_nonsarc = df_nonsarc.rename(
    columns={
        text_source: 'text',
        'photo_path': 'image_path'
    }
)

df_sarcasm.dropna(inplace=True)
df_nonsarc.dropna(inplace=True)

df_sarcasm['image_path'] = df_sarcasm['image_path'].swifter.apply(crop_path)
df_nonsarc['image_path'] = df_nonsarc['image_path'].swifter.apply(crop_path)

df_sarcasm = df_sarcasm[df_sarcasm['topic'].swifter.apply(verify_topics)]
df_nonsarc = df_nonsarc[df_nonsarc['topic'].swifter.apply(verify_topics)]
df_sarcasm['topic'] = df_sarcasm['topic'].swifter.apply(translate_topic)
df_nonsarc['topic'] = df_nonsarc['topic'].swifter.apply(translate_topic)

df_sarcasm = df_sarcasm[df_sarcasm['image_path'].swifter.apply(verify_images)]
df_nonsarc = df_nonsarc[df_nonsarc['image_path'].swifter.apply(verify_images)]
df_sarcasm['id'] = df_sarcasm['image_path'].swifter.apply(get_id)
df_nonsarc['id'] = df_nonsarc['image_path'].swifter.apply(get_id)

df_sarcasm['text'] = df_sarcasm['text'].swifter.apply(remove_whitespace)
df_nonsarc['text'] = df_nonsarc['text'].swifter.apply(remove_whitespace)
df_sarcasm = df_sarcasm[df_sarcasm['text'].swifter.apply(verify_text)]
df_nonsarc = df_nonsarc[df_nonsarc['text'].swifter.apply(verify_text)]
df_sarcasm['text'] = df_sarcasm['text'].swifter.apply(process_text)
df_nonsarc['text'] = df_nonsarc['text'].swifter.apply(process_text)

# return_first_2 = lambda text : return_first_n_sentences(text, n=2)
# df_sarcasm['text'] = df_sarcasm['text'].swifter.apply(return_first_2)
# df_nonsarc['text'] = df_nonsarc['text'].swifter.apply(return_first_2)

return_first_words = lambda text : return_first_n_words(text, n=50)
df_sarcasm['text'] = df_sarcasm['text'].swifter.apply(return_first_words)
df_nonsarc['text'] = df_nonsarc['text'].swifter.apply(return_first_words)


def balance_df(df1: pd.DataFrame, df2: pd.DataFrame, topic: str, 
    _min: Union[int, None] = None):

    df1_topic = df1[df1['topic'] == topic]
    df2_topic = df2[df2['topic'] == topic]
    df1_rest = df1[df1['topic'] != topic]
    df2_rest = df2[df2['topic'] != topic]

    if _min is None:
        _min = min(len(df1_topic), len(df2_topic))

    df1_topic = df1_topic[:_min]
    df2_topic = df2_topic[:_min]

    df1_merged = pd.concat([df1_rest, df1_topic])
    df2_merged = pd.concat([df2_rest, df2_topic])

    print(f'For the topic "{topic}", after balancing there are {2 * _min} articles in total')
    
    return df1_merged, df2_merged, _min

# sort to take the largest articles first
s1 = df_sarcasm.text.str.len().sort_values(ascending=False).index
df_sarcasm = df_sarcasm.reindex(s1)

# sort to take the largest articles first
s2 = df_nonsarc.text.str.len().sort_values(ascending=False).index
df_nonsarc = df_nonsarc.reindex(s2)

# # sort to take the smallest articles first
# s2 = df_nonsarc.text.str.len().sort_values(ascending=True).index
# df_nonsarc = df_nonsarc.reindex(s2)

df_sarcasm, df_nonsarc, _min = balance_df(df_sarcasm, df_nonsarc, 'sports')
df_sarcasm, df_nonsarc, _ = balance_df(df_sarcasm, df_nonsarc, 'politics', _min)
df_sarcasm, df_nonsarc, _ = balance_df(df_sarcasm, df_nonsarc, 'social', _min)

lengths_sarc = np.array([len(text) for text in df_sarcasm['text']])
lengths_nonsarc = np.array([len(text) for text in df_nonsarc['text']])
print(f'Sarcastic text length:')
print(f'\tMAX    - {lengths_sarc.max()}')
print(f'\tMEAN   - {lengths_sarc.mean()}')
print(f'\tMEDIAN - {np.median(lengths_sarc)}')
print(f'Non-sarcastic text length:')
print(f'\tMAX    - {lengths_nonsarc.max()}')
print(f'\tMEAN   - {lengths_nonsarc.mean()}')
print(f'\tMEDIAN - {np.median(lengths_nonsarc)}')

df = pd.concat([df_sarcasm, df_nonsarc])

df.dropna(inplace=True)

if not len(df['id'].unique()) == len(df):
    raise Exception('The ids of the resulting dataset are not all unique')

for text in df_sarcasm['text'].to_list():
    print(text)
    break

df.to_csv(
    args.output_path,
    columns=['id', 'sarcastic', 'topic', 'url', 'image_path', 'text'],
    sep='\t', 
    index=False
)
