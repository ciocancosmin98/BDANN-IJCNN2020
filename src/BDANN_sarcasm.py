import numpy as np
import argparse
import time, os
# import random
import process_twitter as process_data
import copy
import pickle as pickle
from random import sample
import torchvision
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn import metrics
from transformers import AutoModel
import pandas as pd

from process_sarcasm import create_metadata, create_subset, load_subset

lmbd = 1.0

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_np(x):
    return x.data.cpu().numpy()

class GradientReversal(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.lmbd = lmbd
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lmbd

class CNN_Fusion(nn.Module):
    def __init__(self, args):
        super(CNN_Fusion, self).__init__()

        self.args = args
        self.event_num = args.event_num

        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19

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
        for param in vgg_19.parameters():
            param.requires_grad = False

        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

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

    def forward(self, text, image):
        # IMAGE
        image = self.vgg(image)  # [N, 512]
        image = F.relu(self.image_fc1(image))

        last_hidden_state = torch.mean(self.bert_model(text)[0], dim=1, keepdim=False)
        text = F.relu(self.fc2(last_hidden_state))
        text_image = torch.cat((text, image), 1)

        # Fake or real
        class_output = self.class_classifier(text_image)

        # Domain (which Event )
        reverse_feature = self.grad_rev(text_image)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

def load_dataset(args):
    dataset_path = '../ROData/sarcasm_dataset.csv'
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

    train_val_df = df[df.topic.isin(['politics', 'social'])].copy()
    test_df = df[df.topic.isin(['sports'])].copy()

    train_subset = create_subset(
        subset=train_val_df, 
        images_root_path=images_root_path, 
        subset_save_dir=subsets_save_path, 
        name='train',
        metadata=metadata,
        force=False
    )

    test_subset = create_subset(
        subset=test_df, 
        images_root_path=images_root_path, 
        subset_save_dir=subsets_save_path, 
        name='test',
        metadata=metadata,
        force=False
    )

    train_subset.print_stats()
    test_subset.print_stats()

    batch_size = 32
    train_loader = load_subset(train_subset, batch_size, shuffle=True)
    test_loader = load_subset(test_subset, batch_size, shuffle=False)

    return train_loader, test_loader

def main(args):
    train_loader, test_loader = load_dataset(args)

    model = CNN_Fusion(args)
    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 lr=args.learning_rate, weight_decay=0.1)

    best_validate_dir = ''

    # Train the Model
    for epoch in tqdm(range(args.num_epochs)):

        p = float(epoch) / 100
        # lambd = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.001 / (1. + 10 * p) ** 0.75

        optimizer.lr = lr
        # rgs.lambd = lambd
        cost_vector = []
        class_cost_vector = []
        domain_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        vali_cost_vector = []

        for batch in tqdm(train_loader, total=(len(train_loader))):
            text_tokens, images, labels, domains = to_var(batch[0]), \
                to_var(batch[1]), to_var(batch[2]), to_var(batch[3])
            
            optimizer.zero_grad()
            class_outputs, domain_outputs = model(text_tokens, images)

            # Fake or Real loss
            class_loss = criterion(class_outputs, labels)
            # Event Loss
            domain_loss = criterion(domain_outputs, domains)
            #loss = class_loss + domain_loss
            loss = class_loss
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)

            accuracy = (labels == argmax.squeeze()).float().mean()

            class_cost_vector.append(class_loss.item())
            domain_cost_vector.append(domain_loss.item())
            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())

        # model.eval()
        # validate_acc_vector_temp = []
        # for batch in tqdm(test_loader, total=(len(test_loader))):
        #     text_tokens, images, labels, domains = to_var(batch[0]), \
        #         to_var(batch[1]), to_var(batch[2]), to_var(batch[3])
        #     validate_outputs, domain_outputs = model(text_tokens, images)
        #     _, validate_argmax = torch.max(validate_outputs, 1)
        #     vali_loss = criterion(validate_outputs, labels)
        #     validate_accuracy = (labels == validate_argmax.squeeze()).float().mean()
        #     vali_cost_vector.append(vali_loss.item())
        #     validate_acc_vector_temp.append(validate_accuracy.item())
        # validate_acc = np.mean(validate_acc_vector_temp)
        # valid_acc_vector.append(validate_acc)
        # model.train()
        # print('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
        #       % (
        #           epoch + 1, args.num_epochs, np.mean(cost_vector), np.mean(class_cost_vector),
        #           np.mean(domain_cost_vector),
        #           np.mean(acc_vector), validate_acc))

    #model = CNN_Fusion(args)
    #model.load_state_dict(torch.load(best_validate_dir))
    #    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for index, batch in tqdm(enumerate(test_loader), total=(len(test_loader))):
        text_tokens, images, labels, domains = to_var(batch[0]), \
            to_var(batch[1]), to_var(batch[2]), to_var(batch[3])
        test_outputs, domain_outputs = model(text_tokens, images)
        _, test_argmax = torch.max(test_outputs, 1)
        if index == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
            test_true = np.concatenate((test_true, to_np(labels.squeeze())), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')

    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    print("Classification Acc: %.4f, AUC-ROC: %.4f"
          % (test_accuracy, test_aucroc))
    print("Classification report:\n%s\n"
          % (metrics.classification_report(test_true, test_pred)))
    print("Classification confusion matrix:\n%s\n"
          % (test_confusion_matrix))

def parse_arguments():
    parser = argparse.ArgumentParser()

    #parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    # parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    #parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    #parser.add_argument('output_file', type=str, metavar='<output_file>', help='')

    parser.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--text_only', type=bool, default=False, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_epochs', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')

    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)