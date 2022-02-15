import pandas as pd
import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from BDANN_sarcasm import CNN_Fusion
from process_sarcasm import MetaData, create_metadata, create_subset, load_subset

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def get_features(model: CNN_Fusion, loader: DataLoader):
    batch_features = []
    for _, batch in tqdm(enumerate(loader), total=(len(loader))):
        text_tokens = to_var(batch[0])
        images = to_var(batch[1])

        text_features, image_features = model.extract_features(text_tokens, images)
        #batch_features.append(torch.cat((text_features, image_features), 1))
        batch_features.append(image_features)

    return torch.cat(batch_features, 0).cpu().detach().numpy()
    
def main(args):
    dataset_path = '../ROData/sarcasm_dataset_1000_1000_1000.csv'
    metadata_path = '../ROData/metadata.json'
    subsets_save_path = '../ROData/subsets'
    images_root_path = '../ROData/images'

    metadata = create_metadata(
        dataset_path=dataset_path, 
        images_root_path=images_root_path,
        metadata_path=metadata_path,
        force=False
    )

    df = pd.read_csv(dataset_path, delimiter='\t')

    # shuffle rows
    df = df.sample(frac=1, random_state=42)

    n = 1024
    politics_df = df[df.topic.isin(['politics'])].copy()[:n].copy()
    social_df = df[df.topic.isin(['social'])].copy()[:n].copy()
    sports_df = df[df.topic.isin(['sports'])].copy()[:n].copy()

    politics_subset = create_subset(
        subset=politics_df, 
        images_root_path=images_root_path, 
        subset_save_dir=subsets_save_path, 
        name='politics',
        metadata=metadata,
        force=True
    )
    politics_subset.print_stats()
    social_subset = create_subset(
        subset=social_df, 
        images_root_path=images_root_path, 
        subset_save_dir=subsets_save_path, 
        name='social',
        metadata=metadata,
        force=True
    )
    social_subset.print_stats()
    sports_subset = create_subset(
        subset=sports_df, 
        images_root_path=images_root_path, 
        subset_save_dir=subsets_save_path, 
        name='sports',
        metadata=metadata,
        force=True
    )
    sports_subset.print_stats()

    politics_loader = load_subset(politics_subset, 32, True, 8)
    social_loader = load_subset(social_subset, 32, True, 8)
    sports_loader = load_subset(sports_subset, 32, True, 8)

    model = CNN_Fusion(args)
    model.load_state_dict(torch.load(args.model_path))
    if torch.cuda.is_available():
        model.cuda()

    features = []
    features.append(get_features(model, politics_loader))
    features.append(get_features(model, social_loader))
    features.append(get_features(model, sports_loader))

    n_samples = [len(fts) for fts in features]

    features = np.concatenate(features, 0)
    features = (features - features.mean(axis=0)) / (0.00000001 + features.std(axis=0))

    reduced_features = TSNE(n_components=2, learning_rate='auto',
        init='random').fit_transform(features)

    subsets = ['politics'] * n_samples[0] + ['social'] * n_samples[1] + ['sports'] * n_samples[2]

    df = pd.DataFrame(dict(f1=reduced_features[:, 0], f2=reduced_features[:, 1], subset=subsets))

    fig, ax = plt.subplots()
    colors = {'politics':'red', 'social':'orange', 'sports':'blue'}

    grouped = df.groupby('subset')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='f1', y='f2', s=7, label=key, color=colors[key])

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.title('Non-adapted image features')
    plt.savefig("output_images_0.5.png")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lmbd', type=float, default=1, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_epochs', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--model_path', type=str, required=True, help='The path to the model to be evaluated.')

    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()
    args.text_only = False
    args.images_only = False
    main(args)