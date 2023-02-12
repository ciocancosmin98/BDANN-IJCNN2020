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

        if args.freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        #self.bertModel = bert_model

        self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        # params = []
        # for param in vgg_19.parameters():
        #     param.requires_grad = False

        #     params.append(param)

        # params[-1].requires_grad = True # retrain last dense layer's bias
        # params[-2].requires_grad = True # retrain last dense layer's weights    

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

    def predict(
        self,
        text,
        images
    ):

        # IMAGE
        if self.text_only:
            image_features = torch.zeros((text.shape[0], self.hidden_size), device='cuda:0')
        else:
            image_features = self.vgg(images)
            image_features = F.relu(self.image_fc1(image_features))

        if self.images_only:
            text_features = torch.zeros((text.shape[0], self.hidden_size), device='cuda:0')
        else:
            last_hidden_state = torch.mean(
                self.bert_model(text)[0], 
                dim=1, 
                keepdim=False
            )
            text_features = F.relu(self.fc2(last_hidden_state))
        
        multimodal_features = torch.cat((text_features, image_features), 1)

        # Sarcastic or not
        class_output = self.class_classifier(multimodal_features)

        # Domain (which Event )
        # reverse_feature = self.grad_rev(multimodal_features_all, lmbd)
        # domain_output = self.domain_classifier(reverse_feature)

        return class_output

    def forward(
        self,
        src_text, 
        tgt_text,
        src_images,
        tgt_images,
        lmbd = 0.5
    ):
        bs_src = len(src_text)
        bs_tgt = len(tgt_text)
        bs_total = bs_src + bs_tgt

        # IMAGE
        if self.text_only:
            image_features = torch.zeros((bs_total, self.hidden_size), device='cuda:0')
        else:
            all_images = torch.cat((src_images, tgt_images), axis=0)
            image_features = self.vgg(all_images)
            image_features = F.relu(self.image_fc1(image_features))

        if self.images_only:
            text_features = torch.zeros((bs_total, self.hidden_size), device='cuda:0')
        else:
            all_text = torch.cat((src_text, tgt_text), axis=0)
            last_hidden_state = torch.mean(
                self.bert_model(all_text)[0], 
                dim=1, 
                keepdim=False
            )
            text_features = F.relu(self.fc2(last_hidden_state))
        
        multimodal_features_all = torch.cat((text_features, image_features), 1)
        multimodal_features_source = multimodal_features_all[:bs_src]

        # Sarcastic or not
        class_output = self.class_classifier(multimodal_features_source)

        # Domain (which Event )
        reverse_feature = self.grad_rev(multimodal_features_all, lmbd)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

def load_dataset(
    train_path: str,
    dev_path: str,
    test_path: str,
    subsets_save_path: str, 
    images_root_path: str,
    metadata: MetaData, 
    source_topic: str, 
    target_topic: str, 
    batch_size = 8, 
    verbose=False
):

    df_train = pd.read_csv(train_path, delimiter='\t')
    df_dev = pd.read_csv(dev_path, delimiter='\t')
    df_test = pd.read_csv(test_path, delimiter='\t')

    if not source_topic is None:
        assert source_topic in metadata.domain_mapping

        source_train_df = df_train[df_train.topic.isin([source_topic])].copy()
        source_dev_df = df_dev[df_dev.topic.isin([source_topic])].copy()

        # assert len(source_df) > 0

        source_train = create_subset(
            subset=source_train_df, 
            images_root_path=images_root_path, 
            subset_save_dir=subsets_save_path, 
            name=f'source_{source_topic}_train',
            metadata=metadata,
            force=True
        )
        source_valid = create_subset(
            subset=source_dev_df, 
            images_root_path=images_root_path, 
            subset_save_dir=subsets_save_path, 
            name=f'source_{source_topic}_valid',
            metadata=metadata,
            force=True
        )
        # if verbose:
        #     source_subset.print_stats()
        
        # source_train, source_valid = \
        #     train_validate_split(source_subset, train_ratio=0.9, seed=29)

        src_train_loader = load_subset(source_train, batch_size, shuffle=True)
        src_valid_loader = load_subset(source_valid, batch_size, shuffle=False)
    else:
        src_train_loader = None
        src_valid_loader = None

    if not target_topic is None:
        assert target_topic in metadata.domain_mapping

        # target_df = df[df.topic.isin([target_topic])].copy()
        target_train_df = df_train[df_train.topic.isin([target_topic])].copy()
        target_test_df = df_test[df_test.topic.isin([target_topic])].copy()

        # assert len(target_df) > 0

        target_train = create_subset(
            subset=target_train_df, 
            images_root_path=images_root_path, 
            subset_save_dir=subsets_save_path, 
            name=f'target_{target_topic}_valid',
            metadata=metadata,
            force=True
        )
        target_test = create_subset(
            subset=target_test_df, 
            images_root_path=images_root_path, 
            subset_save_dir=subsets_save_path, 
            name=f'target_{target_topic}_valid',
            metadata=metadata,
            force=True
        )
        # if verbose:
        #     target_subset.print_stats()
        
        # target_train, target_valid = \
        #     train_validate_split(target_subset, train_ratio=0.5, seed=29)

        tgt_train_loader = load_subset(target_train, batch_size, shuffle=True)
        tgt_test_loader = load_subset(target_test, batch_size, shuffle=False)
    else:
        tgt_train_loader = None
        tgt_test_loader = None

    return (src_train_loader, src_valid_loader), (tgt_train_loader, tgt_test_loader)

def evaluate_loop(
    model: CNN_Fusion, 
    loader: DataLoader, 
    save_predictions: Union[None, str] = None
):
    total_label_loss = 0
    total_domain_loss = 0
    n = 0

    for index, batch in tqdm(enumerate(loader), total=(len(loader))):
        text_tokens = to_var(batch[0])
        images = to_var(batch[1])
        labels = to_var(batch[2])
        # domains = to_var(batch[3])
        ids_batch: List[str] = list(batch[4])

        with torch.no_grad():
            label_outputs = model.predict(text_tokens, images)

            criterion = nn.CrossEntropyLoss()
            total_label_loss += criterion(label_outputs, labels).item()
            # total_domain_loss += criterion(domain_outputs, domains).item()
            n += len(label_outputs)


        # apply argmax to select the class with the largest confidence value
        _, label_argmax = torch.max(label_outputs, 1)
        # _, domain_argmax = torch.max(domain_outputs, 1)

        if index == 0:
            label_pred = to_np(label_argmax.squeeze())
            label_true = to_np(labels.squeeze())
            ids_all = ids_batch
            # domain_pred = to_np(domain_argmax.squeeze())
            # domain_true = to_np(domains.squeeze())
        else:
            label_pred = np.concatenate((label_pred, to_np(label_argmax.squeeze())), axis=0)
            label_true = np.concatenate((label_true, to_np(labels.squeeze())), axis=0)
            ids_all.extend(ids_batch)
            # domain_pred = np.concatenate((domain_pred, to_np(domain_argmax.squeeze())), axis=0)
            # domain_true = np.concatenate((domain_true, to_np(domains.squeeze())), axis=0)

    preds = {
        'label': {
            'pred': label_pred,
            'true': label_true,
            'ids': ids_all
        },
        'domain': {
            # 'pred': domain_pred,
            # 'true': domain_true
        }
    }

    if save_predictions is not None:
        data = {
            'id': ids_all,
            'predicted': list(label_pred),
            'ground_truth': list(label_true),
        }
        results_df = pd.DataFrame.from_dict(data)
        results_df.to_csv(save_predictions, index=False, sep='\t')

    results = {name: {} for name in preds}
    results['label']['loss'] = total_label_loss / n
    results['domain']['loss'] = total_domain_loss / n

    for pred_name in ['label']:

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

def train_loop(
    model: CNN_Fusion, 
    src_train_loader: DataLoader,
    tgt_train_loader: DataLoader, 
    valid_loader: DataLoader, 
    save_dir: str, 
    n_epochs = 10, 
    lr = 0.001, 
    domain_adaptation = True, 
    verbose = False
):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [
            {'params': model.bert_model.parameters(), 'lr': lr * 1e-3, 'weight_decay': 0},
            {'params': model.class_classifier.parameters(), 'lr': lr},
            {'params': model.domain_classifier.parameters(), 'lr': lr},
            {'params': model.vgg.parameters(), 'lr': lr * 1e-3, 'weight_decay': 0},
            {'params': model.fc2.parameters(), 'lr': lr},
            {'params': model.image_fc1.parameters(), 'lr': lr}
        ],
        #filter(lambda p: p.requires_grad, list(model.parameters())),
        lr=lr, 
        weight_decay=0.1
    )

    best_valid_loss = 1e8

    epoch_steps = len(src_train_loader)
    total_steps = n_epochs * epoch_steps

    for epoch in tqdm(range(n_epochs)):

        p = float(epoch) / n_epochs

        optimizer.lr = 0.001 / (1. + 10 * p) ** 0.75
        # rgs.lambd = lambd
        cost_vector = []
        class_cost_vector = []
        domain_cost_vector = []
        acc_vector = []

        tgt_train_iter = iter(tgt_train_loader)

        for index, src_batch in enumerate(tqdm(src_train_loader, total=(len(src_train_loader)))):
            src_text_tokens, src_images, src_labels = to_var(src_batch[0]), \
                to_var(src_batch[1]), to_var(src_batch[2])

            try:
                tgt_batch = next(tgt_train_iter)
            except:
                tgt_train_iter = iter(tgt_train_loader)
                tgt_batch = next(tgt_train_iter)

            tgt_text_tokens, tgt_images = to_var(tgt_batch[0]), to_var(tgt_batch[1])
            
            step = epoch * epoch_steps + index
            lmbd = args.lmbd * (step / total_steps)

            assert isinstance(lmbd, float) and lmbd >= 0.0 and lmbd <= 1.0

            domains = torch.cat((
                torch.zeros(len(src_text_tokens), dtype=torch.int64, device='cuda:0'), 
                torch.ones(len(tgt_text_tokens), dtype=torch.int64, device='cuda:0')
            ))

            optimizer.zero_grad()
            class_outputs, domain_outputs = model(
                src_text_tokens, 
                tgt_text_tokens,
                src_images,
                tgt_images,
                lmbd
            )

            class_loss = criterion(class_outputs, src_labels)
            domain_loss = criterion(domain_outputs, domains)

            if domain_adaptation:
                loss = class_loss + domain_loss
            else:
                loss = class_loss

            loss.backward()
            optimizer.step()

            _, argmax = torch.max(class_outputs, 1)
            accuracy = (src_labels == argmax.squeeze()).float().mean()

            class_cost_vector.append(class_loss.item())
            domain_cost_vector.append(domain_loss.item())
            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())

        model.eval()
        # results_train = evaluate_loop(model, src_train_loader)
        results_valid = evaluate_loop(model, valid_loader)
        model.train()

        # print("Train Acc: %.4f"
        #     % (results_train['label']['accuracy']))
        # print("Train loss:\n%s\n"
        #     % (results_train['label']['loss']))
        # print("Train report:\n%s\n"
        #     % (results_train['label']['report']))
        # print("Train confusion matrix:\n%s\n"
        #     % (results_train['label']['confusion_matrix']))

        print("Val Acc: %.4f"
            % (results_valid['label']['accuracy']))
        print("Val loss:\n%s\n"
            % (results_valid['label']['loss']))
        print("Val report:\n%s\n"
            % (results_valid['label']['report']))
        print("Val confusion matrix:\n%s\n"
            % (results_valid['label']['confusion_matrix']))

        best = False
        curr_loss = results_valid['label']['loss'] 
        if curr_loss < best_valid_loss:
            best_valid_loss = curr_loss
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
                    results_valid['label']['accuracy']
                )
            )
            print("Domain report:\n%s\n"
                % (results_valid['domain']['report']))
            print("Domain confusion matrix:\n%s\n"
                % (results_valid['domain']['confusion_matrix']))

    return best_model_path

def main(args):
    accuracies = []
    f1_scores = []
    recall_scores = []
    precision_scores = []

    # dataset_path = '../ROData/musaro_processed.tsv'

    train_path = '../ROData/musaro_train_processed.tsv'
    dev_path = '../ROData/musaro_dev_processed.tsv'
    test_path = '../ROData/musaro_test_processed.tsv'

    # dataset_path = '../ROData/sarcasm_title_anonimized_1000.csv'
    metadata_path = '../ROData/metadata.json'
    subsets_save_path = '../ROData/subsets'
    images_root_path = '../ROData/musaro_news/'

    metadata = create_metadata(
        dataset_path=train_path, 
        images_root_path=images_root_path,
        metadata_path=metadata_path,
        force=False
    )

    for run_index in tqdm(range(args.n_runs)):
        src_loaders, tgt_loaders = load_dataset(
            train_path=train_path, 
            dev_path=dev_path, 
            test_path=test_path, 
            subsets_save_path=subsets_save_path, 
            images_root_path=images_root_path, 
            metadata=metadata, 
            source_topic=args.source_topic, 
            target_topic=args.target_topic
        )
        src_train_loader, src_valid_loader = src_loaders
        tgt_train_loader, tgt_test_loader = tgt_loaders

        if len(args.evaluate) == 0:
            model = CNN_Fusion(args)
            if torch.cuda.is_available():
                model.cuda()
            model_path = train_loop(
                model=model,
                src_train_loader=src_train_loader,
                tgt_train_loader=tgt_train_loader, 
                valid_loader=src_valid_loader,
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
        results = evaluate_loop(model, tgt_test_loader, args.save_predictions)

        print("Test Acc: %.4f"
            % (results['label']['accuracy']))
        print("Test loss:\n%s\n"
            % (results['label']['loss']))
        print("Test report:\n%s\n"
            % (results['label']['report']))
        print("Test confusion matrix:\n%s\n"
            % (results['label']['confusion_matrix']))

        if len(args.evaluate) != 0:
            print(f"Results available at {args.save_predictions}")
            exit()

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
    parser.add_argument('--freeze_bert', action='store_true', help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lmbd', type=float, default=1, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--num_epochs', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--save_dir', type=str, default='../models/testrun')
    parser.add_argument('--text_only', action='store_true')
    parser.add_argument('--images_only', action='store_true')
    parser.add_argument('--no_domain_adaptation', action='store_true')
    parser.add_argument('--source_topic', type=str, required=True)
    parser.add_argument('--target_topic', type=str, required=True)
    parser.add_argument('--save_predictions', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--evaluate', type=str, default='', help='The path to the model to be evaluated.')
    parser.add_argument('--n_runs', type=int, default=1)

    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')

    args = parser.parse_args()

    assert not (args.text_only and args.images_only)

    if len(args.evaluate) != 0:
        # evaluation only mode
        assert os.path.exists(args.evaluate)

    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)