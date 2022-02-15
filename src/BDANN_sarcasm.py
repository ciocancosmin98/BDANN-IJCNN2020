import json
import random
from typing import List, Union
import numpy as np
import argparse
import os
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics
from transformers import AutoModel
import pandas as pd

from process_sarcasm import MetaData, create_metadata, create_subset, load_subset, train_validate_split, MappingDataset

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_np(x):
    return x.data.cpu().numpy()

class GradientReversal(Function):

    @staticmethod
    def forward(ctx, x, lmbd):
        ctx.lmbd = lmbd
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lmbd, None

class CNN_Fusion(nn.Module):
    def __init__(self, args):
        super(CNN_Fusion, self).__init__()

        self.args = args
        if not hasattr(args, 'train_topics') or args.train_topics is None:
            self.event_num = 2
        else:
            self.event_num = len(args.train_topics)

        self.text_only = args.text_only
        self.images_only = args.images_only

        if self.text_only and args.verbose:
            print('Running only on text')
        if self.images_only and args.verbose:
            print('Running on images only')

        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim

        # bert
        self.bert_model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1")
        self.bert_hidden_size = args.bert_hidden_dim
        self.fc2 = nn.Linear(self.bert_hidden_size, self.hidden_size)

        for param in self.bert_model.parameters():
            param.requires_grad = False
        #self.bertModel = bert_model

        self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        params = []
        for param in vgg_19.parameters():
            param.requires_grad = False

            params.append(param)

        params[-1].requires_grad = True # retrain last dense layer's bias
        params[-2].requires_grad = True # retrain last dense layer's weights    

        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        # mixed features
        # self.fc3 = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)

        # Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        # Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(2 * self.hidden_size, self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

        self.grad_rev = GradientReversal.apply

    def extract_features(self, text, image):
        image_features = self.vgg(image) 
        image_features = F.relu(self.image_fc1(image_features))
            
        last_hidden_state = torch.mean(self.bert_model(text)[0], dim=1, keepdim=False)
        text_features = F.relu(self.fc2(last_hidden_state))

        return text_features, image_features


    def forward(self, text, image):
        # IMAGE
        if self.text_only:
            image = torch.zeros((text.shape[0], self.hidden_size), device='cuda:0')
        else:
            image = self.vgg(image)  # [N, 512]
            image = F.relu(self.image_fc1(image))

        if self.images_only:
            text = torch.zeros((text.shape[0], self.hidden_size), device='cuda:0')
        else:
            last_hidden_state = torch.mean(self.bert_model(text)[0], dim=1, keepdim=False)
            text = F.relu(self.fc2(last_hidden_state))
        
        text_image = torch.cat((text, image), 1)
        #mixed_features = F.relu(self.fc3(text_image))

        # Sarcastic or not
        class_output = self.class_classifier(text_image)
        #class_output = self.class_classifier(mixed_features)

        # Domain (which Event )
        reverse_feature = self.grad_rev(text_image, args.lmbd)
        #reverse_feature = self.grad_rev(mixed_features, args.lmbd)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

def load_dataset(dataset_path: str, subsets_save_path: str, images_root_path: str,
    metadata: MetaData, test_topics: Union[List[str], None], 
    train_topics: Union[List[str], None], batch_size = 32, verbose=False):

    df = pd.read_csv(dataset_path, delimiter='\t')

    if not train_topics is None:
        for topic in train_topics:
            assert topic in metadata.domain_mapping

        train_val_df = df[df.topic.isin(train_topics)].copy()

        assert len(train_val_df) > 0

        train_topics.sort()
        subset_name = ('-').join(train_topics)

        train_val_subset = create_subset(
            subset=train_val_df, 
            images_root_path=images_root_path, 
            subset_save_dir=subsets_save_path, 
            name=subset_name,
            metadata=metadata,
            force=True
        )
        if verbose:
            train_val_subset.print_stats()
        
        train_subset, valid_subset = \
            train_validate_split(train_val_subset, train_ratio=0.9, seed=29)

        train_loader = load_subset(train_subset, batch_size, shuffle=True)
        valid_loader = load_subset(valid_subset, batch_size, shuffle=False)
    else:
        train_loader = None
        valid_loader = None

    if not test_topics is None:
        for topic in test_topics:
            assert topic in metadata.domain_mapping

        test_df = df[df.topic.isin(test_topics)].copy()

        assert len(test_df) > 0

        test_topics.sort()
        subset_name = ('-').join(test_topics)

        test_subset = create_subset(
            subset=test_df, 
            images_root_path=images_root_path, 
            subset_save_dir=subsets_save_path, 
            name=subset_name,
            metadata=metadata,
            force=True
        )
        if verbose:
            test_subset.print_stats()

        # delete me #####################
        # random_perm = list(range(len(test_subset)))
        # random.shuffle(random_perm)
        # test_subset = MappingDataset(test_subset, random_perm)
        #################################

        test_loader = load_subset(test_subset, batch_size, shuffle=False)
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader

def evaluate_loop(model: CNN_Fusion, loader: DataLoader):
    for index, batch in tqdm(enumerate(loader), total=(len(loader))):
        text_tokens = to_var(batch[0])
        images = to_var(batch[1])
        labels = to_var(batch[2])
        domains = to_var(batch[3])

        label_outputs, domain_outputs = model(text_tokens, images)

        # apply argmax to select the class with the largest confidence value
        _, label_argmax = torch.max(label_outputs, 1)
        _, domain_argmax = torch.max(domain_outputs, 1)

        if index == 0:
            label_pred = to_np(label_argmax.squeeze())
            label_true = to_np(labels.squeeze())
            domain_pred = to_np(domain_argmax.squeeze())
            domain_true = to_np(domains.squeeze())
        else:
            label_pred = np.concatenate((label_pred, to_np(label_argmax.squeeze())), axis=0)
            label_true = np.concatenate((label_true, to_np(labels.squeeze())), axis=0)
            domain_pred = np.concatenate((domain_pred, to_np(domain_argmax.squeeze())), axis=0)
            domain_true = np.concatenate((domain_true, to_np(domains.squeeze())), axis=0)

    preds = {
        'label': {
            'pred': label_pred,
            'true': label_true
        },
        'domain': {
            'pred': domain_pred,
            'true': domain_true
        }
    }

    results = {name: {} for name in preds}

    for pred_name in preds:

        pred = preds[pred_name]['pred']
        true = preds[pred_name]['true']

        # skip calculating metrics if
        if true.min() == true.max():
            continue

        results[pred_name]['accuracy'] = metrics.accuracy_score(true, pred)
        results[pred_name]['f1_score'] = metrics.f1_score(true, pred, average=None)
        results[pred_name]['precision'] = metrics.precision_score(true, pred, average=None)
        results[pred_name]['recall'] = metrics.recall_score(true, pred, average=None)

        results[pred_name]['confusion_matrix'] = metrics.confusion_matrix(true, pred)
        results[pred_name]['report'] = metrics.classification_report(true, pred)

    return results

def train_loop(model: CNN_Fusion, train_loader: DataLoader, 
    valid_loader: DataLoader, save_dir: str, n_epochs = 10, lr = 0.001, 
    domain_adaptation = True, verbose = False):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, list(model.parameters())),
        lr=lr, 
        weight_decay=0.1
    )

    best_valid_acc = 0.0

    for epoch in tqdm(range(n_epochs)):

        p = float(epoch) / n_epochs

        optimizer.lr = 0.001 / (1. + 10 * p) ** 0.75
        # rgs.lambd = lambd
        cost_vector = []
        class_cost_vector = []
        domain_cost_vector = []
        acc_vector = []

        for batch in tqdm(train_loader, total=(len(train_loader))):
            text_tokens, images, labels, domains = to_var(batch[0]), \
                to_var(batch[1]), to_var(batch[2]), to_var(batch[3])
            
            optimizer.zero_grad()
            class_outputs, domain_outputs = model(text_tokens, images)

            class_loss = criterion(class_outputs, labels)
            domain_loss = criterion(domain_outputs, domains)

            if domain_adaptation:
                loss = class_loss + domain_loss
            else:
                loss = class_loss

            loss.backward()
            optimizer.step()

            _, argmax = torch.max(class_outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().mean()

            class_cost_vector.append(class_loss.item())
            domain_cost_vector.append(domain_loss.item())
            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())

        model.eval()
        results = evaluate_loop(model, valid_loader)
        model.train()

        best = False
        if results['label']['accuracy'] > best_valid_acc:
            best_valid_acc = results['label']['accuracy']
            best = True

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_name = str(epoch + 1)
        if best:
            model_name = model_name + '-best'
            best_model_path = os.path.join(save_dir, model_name)

        torch.save(model.state_dict(), os.path.join(save_dir, model_name))

        if verbose:
            print('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.' % \
                (
                    epoch + 1, 
                    args.num_epochs, 
                    np.mean(cost_vector), 
                    np.mean(class_cost_vector),
                    np.mean(domain_cost_vector),
                    np.mean(acc_vector), 
                    results['label']['accuracy']
                )
            )
            print("Domain report:\n%s\n"
                % (results['domain']['report']))
            print("Domain confusion matrix:\n%s\n"
                % (results['domain']['confusion_matrix']))

    return best_model_path

def main(args):
    accuracies = []
    f1_scores = []
    recall_scores = []
    precision_scores = []

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

    for run_index in tqdm(range(args.n_runs)):
        train_loader, valid_loader, test_loader = load_dataset(
            dataset_path=dataset_path, 
            subsets_save_path=subsets_save_path, 
            images_root_path=images_root_path, 
            metadata=metadata, 
            train_topics=args.train_topics, 
            test_topics=args.test_topics
        )

        if len(args.evaluate) == 0:
            model = CNN_Fusion(args)
            if torch.cuda.is_available():
                model.cuda()
            model_path = train_loop(
                model=model, 
                train_loader=train_loader, 
                valid_loader=valid_loader,
                save_dir=os.path.join(args.save_dir, f'run_{run_index:02}'), 
                n_epochs=args.num_epochs, 
                lr=args.learning_rate,
                domain_adaptation=(not args.no_domain_adaptation)
            )
        else:
            model_path = args.evaluate

        model = CNN_Fusion(args)
        model.load_state_dict(torch.load(model_path))
        if torch.cuda.is_available():
            model.cuda()

        model.eval()
        results = evaluate_loop(model, test_loader)

        print("Classification Acc: %.4f"
            % (results['label']['accuracy']))
        print("Classification report:\n%s\n"
            % (results['label']['report']))
        print("Classification confusion matrix:\n%s\n"
            % (results['label']['confusion_matrix']))

        results = results['label']

        accuracies.append(np.expand_dims(results['accuracy'], axis=0))
        f1_scores.append(np.expand_dims(results['f1_score'], axis=0))
        precision_scores.append(np.expand_dims(results['precision'], axis=0))
        recall_scores.append(np.expand_dims(results['recall'], axis=0))
    
    accuracies = np.concatenate(accuracies, axis=0)
    f1_scores = np.concatenate(f1_scores, axis=0)
    precision_scores = np.concatenate(precision_scores, axis=0)
    recall_scores = np.concatenate(recall_scores, axis=0)

    labels = [f'sarcastic_{metadata.reverse_label_mapping()[index]}' for index in [0, 1]]

    data = {
        'run_index': list(range(args.n_runs)),
        'accuracy': accuracies.tolist(),
        f'f1_scores_{labels[0]}': f1_scores[:, 0].tolist(),
        f'f1_scores_{labels[1]}': f1_scores[:, 1].tolist(),
        f'precision_scores_{labels[0]}': precision_scores[:, 0].tolist(),
        f'precision_scores_{labels[1]}': precision_scores[:, 1].tolist(),
        f'recall_scores_{labels[0]}': recall_scores[:, 0].tolist(),
        f'recall_scores_{labels[1]}': recall_scores[:, 1].tolist(),
    }

    df = pd.DataFrame.from_dict(data)

    print(df.head())
    print(df.mean())

    df.to_csv(os.path.join(args.save_dir, 'results.tsv'), sep='\t', index=False)


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
    parser.add_argument('--save_dir', type=str, default='../models/testrun')
    parser.add_argument('--text_only', action='store_true')
    parser.add_argument('--images_only', action='store_true')
    parser.add_argument('--no_domain_adaptation', action='store_true')
    parser.add_argument('--train_topics', nargs='+')
    parser.add_argument('--test_topics', nargs='+')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--evaluate', type=str, default='', help='The path to the model to be evaluated.')
    parser.add_argument('--n_runs', type=int, default=1)

    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')

    args = parser.parse_args()

    assert not (args.text_only and args.images_only)

    if len(args.evaluate) != 0:
        # evaluation only mode
        assert os.path.exists(args.evaluate)
        assert len(args.test_topics) > 0
    else:
        # train mode
        assert len(args.train_topics) > 0
        assert len(args.test_topics) > 0

    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)