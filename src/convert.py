import pandas as pd
import swifter
import numpy as np
from PIL import Image
import os

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

def verify_text(text: str):
    if not isinstance(text, str):
        return False
    if len(text) == 0:
        return False
    return True

def verify_topics(topic: str):
    if topic in translation_dict:
        return True
    return False

def verify_images(image_path: str):
    _path = os.path.abspath(os.path.join('./images', image_path))
    
    try:
        Image.open(_path).convert('RGB')
    except:
        return False

    return True

df_sarcasm = pd.read_csv('../ROData/tnr-all-articles-unique-ids.csv', low_memory=False)
df_nonsarc = pd.read_csv('../ROData/non-sarcasm-unique-ids-tags-updated.csv', low_memory=False)
df_sarcasm['sarcastic'] = 'yes'
df_nonsarc['sarcastic'] = 'no'

# select the columns to be kept from both .csv files
df_sarcasm = df_sarcasm[
    ['sarcastic', 'topic', 'photo_path', 'url', 'text']
]
df_nonsarc = df_nonsarc[
    ['sarcastic', 'category', 'photo_path', 'website', 'content']
]

# rename the columns for compatibility
df_sarcasm = df_sarcasm.rename(
    columns={
        'photo_path': 'image_path'
    }
)
df_nonsarc = df_nonsarc.rename(
    columns={
        'content': 'text', 
        'category': 'topic', 
        'photo_path': 'image_path',
        'website': 'url'
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

df = pd.concat([df_sarcasm, df_nonsarc])

assert len(df['id'].unique()) == len(df)

df.dropna(inplace=True)

lengths = np.array([len(text) for text in df_sarcasm['text']])
print(lengths.mean(), np.median(lengths))
lengths = np.array([len(text) for text in df_nonsarc['text']])
print(lengths.mean(), np.median(lengths))

print(len(df), len(df_sarcasm), len(df_nonsarc))

df.to_csv(
    '../ROData/sarcasm_dataset.csv',
    columns=['id', 'sarcastic', 'topic', 'url', 'image_path', 'text'],
    sep='\t', 
    index=False
)
