import torch
import torch.nn as nn
import os
import numpy as np
from datasets import data_loader
from models import BertRNN
from sklearn.metrics import accuracy_score
import copy
import os
import yaml
import logging
import random
from logging import StreamHandler, FileHandler
from transformers import get_linear_schedule_with_warmup
import sys
from transformers import AdamW
from tqdm import tqdm


logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_logger(log_filename):
    handler1 = StreamHandler(stream=sys.stdout)
    handler2 = FileHandler(filename=log_filename, mode='a', delay=False)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[handler1, handler2]
    )


class Engine:
    def __init__(self, args):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        os.makedirs('ckp', exist_ok=True)
        set_seed(args.seed)
        init_logger(args.log_filename)
        logger.info('=' * 20 + 'Job start !' + '=' * 20)
        train_loader = data_loader(corpus=args.corpus,
                                   phase='train',
                                   batch_size=args.batch_size,
                                   chunk_size=args.chunk_size,
                                   shuffle=True,
                                   max_length=args.max_length) if args.mode != 'inference' else None
        val_loader = data_loader(corpus=args.corpus,
                                 phase='val',
                                 batch_size=args.batch_size_val,
                                 chunk_size=args.chunk_size,
                                 max_length=args.max_length) if args.mode != 'inference' else None
        test_loader = data_loader(corpus=args.corpus,
                                  phase='test',
                                  batch_size=args.batch_size_val,
                                  chunk_size=args.chunk_size,
                                  max_length=args.max_length)


        if torch.cuda.device_count() > 0:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")

        logger.info('Initializing model....')
        logger.info(f'{args.nfinetune}')
        model = BertRNN(nlayer=args.nlayer,
                        nclass=args.nclass,
                        dropout=args.dropout,
                        nfinetune=args.nfinetune,
                        speaker_info=args.speaker_info,
                        topic_info=args.topic_info,
                        emb_batch=args.emb_batch,
                        )

        # model = nn.DataParallel(model)
        model.to(device)
        params = model.parameters()

        optimizer = AdamW(params, lr=args.lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args

        total_step = len(train_loader) * args.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_step * args.warmup_rate),
            num_training_steps=total_step
        )

    def train(self):
        best_epoch = 0
        best_epoch_acc = 0
        best_epoch_test_acc = 0
        best_acc = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.args.epochs):
            logger.info(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
            loss = self.train_epoch(epoch, self.args.epochs)
            acc = self.eval('valid')
            test_acc = self.eval('test')
            if acc > best_epoch_acc:
                best_epoch = epoch
                best_epoch_acc = acc
                best_epoch_test_acc = test_acc
                best_state_dict = copy.deepcopy(self.model.state_dict())
            if test_acc > best_acc:
                best_acc = test_acc
            logger.info(f'Epoch {epoch + 1}\tTrain Loss: {loss:.3f}\tValid Acc: {acc:.3f}\tTest Acc: {test_acc:.3f}\n'
                  f'Best Epoch: {best_epoch + 1}\tBest Epoch Val Acc: {best_epoch_acc:.3f}\t'
                  f'Best Epoch Test Acc: {best_epoch_test_acc:.3f}, Best Test Acc: {best_acc:.3f}\n')
            if epoch - best_epoch >= 10000:
                break

        logger.info('Saving the best checkpoint....')
        torch.save(best_state_dict, f"ckp/model_{self.args.corpus}.pt")
        self.model.load_state_dict(best_state_dict)
        acc = self.eval('test')
        logger.info(f'Test Acc: {acc:.3f}')

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(tqdm(self.train_loader, desc=f'Train epoch {epoch + 1} / {total_epochs}', ncols=100)):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            chunk_lens = batch['chunk_lens']
            speaker_ids = batch['speaker_ids'].to(self.device)
            topic_labels = batch['topic_labels'].to(self.device)
            chunk_attention_mask = batch['chunk_attention_mask'].to(self.device)
            outputs = self.model(input_ids, attention_mask, chunk_lens, speaker_ids,
                                 topic_labels, chunk_attention_mask)
            labels = labels.reshape(-1)
            loss_act = self.criterion(outputs, labels)
            loss = loss_act
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            # interval = max(len(self.train_loader) // 20, 1)
            # if i % interval == 0 or i == len(self.train_loader) - 1:
            #     logger.info(f'Batch: {i + 1}/{len(self.train_loader)}\tloss: {loss.item():.3f}')
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)

    def eval(self, mode='valid', inference=False):
        self.model.eval()
        y_pred = []
        y_true = []
        loader = self.val_loader if mode == 'valid' else self.test_loader
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader, desc=mode, ncols=100)):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                chunk_lens = batch['chunk_lens']
                speaker_ids = batch['speaker_ids'].to(self.device)
                topic_labels = batch['topic_labels'].to(self.device)
                chunk_attention_mask = batch['chunk_attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask, chunk_lens, speaker_ids, topic_labels, chunk_attention_mask)
                y_pred.append(outputs.detach().to('cpu').argmax(dim=1).numpy())
                labels = labels.reshape(-1)
                y_true.append(labels.detach().to('cpu').numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        mask = y_true != -1
        acc = accuracy_score(y_true[mask], y_pred[mask])

        if inference:
            import pickle
            pickle.dump(y_pred[mask].tolist(), open('preds_on_new.pkl', 'wb'))

        return acc

    def inference(self):
        ## using the trained model to inference on a new unseen dataset

        # load the saved checkpoint
        # change the model name to whatever the checkpoint is named
        self.model.load_state_dict(torch.load('ckp/model.pt'))

        # make predictions
        self.eval(val=False, inference=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config.yaml', help='config file path')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        dict_load = yaml.load(f, Loader=yaml.FullLoader)
    
    args.corpus = dict_load['corpus']
    args.mode = dict_load['mode']
    args.nclass = dict_load['nclass']
    args.batch_size = dict_load['batch_size']
    args.batch_size_val = dict_load['batch_size_val']
    args.emb_batch = dict_load['emb_batch']
    args.epochs = dict_load['epochs']
    args.lr = float(dict_load['lr'])
    args.nlayer = dict_load['nlayer']
    args.chunk_size = dict_load['chunk_size']
    args.dropout = dict_load['dropout']
    args.speaker_info = dict_load['speaker_info']
    args.topic_info = dict_load['topic_info']
    args.nfinetune = dict_load['nfinetune']
    args.seed = dict_load['seed']
    args.warmup_rate = float(dict_load['warmup_rate'])
    args.max_length = dict_load['max_length']

    config_path = '/'.join(args.config_file.split('/')[: -1])
    args.log_filename = os.path.join(config_path, args.corpus + '.log')

    logger.info(f'{args}')
    engine = Engine(args)
    if args.mode == 'train':
        engine.train()
    else:
        engine.inference()
