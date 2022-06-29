import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import logging
import re
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
### Dialogue act label encoding, SWDA
# {'qw^d': 0, '^2': 1, 'b^m': 2, 'qy^d': 3, '^h': 4, 'bk': 5, 'b': 6, 'fa': 7, 'sd': 8, 'fo_o_fw_"_by_bc': 9,
#              'ad': 10, 'ba': 11, 'ng': 12, 't1': 13, 'bd': 14, 'qh': 15, 'br': 16, 'qo': 17, 'nn': 18, 'arp_nd': 19,
#              'fp': 20, 'aap_am': 21, 'oo_co_cc': 22, 'h': 23, 'qrr': 24, 'na': 25, 'x': 26, 'bh': 27, 'fc': 28,
#              'aa': 29, 't3': 30, 'no': 31, '%': 32, '^g': 33, 'qy': 34, 'sv': 35, 'ft': 36, '^q': 37, 'bf': 38,
#              'qw': 39, 'ny': 40, 'ar': 41, '+': 42}

### Topic label encoding, SWDA
# {'CARE OF THE ELDERLY': 0, 'HOBBIES AND CRAFTS': 1, 'WEATHER CLIMATE': 2, 'PETS': 3,
#              'CHOOSING A COLLEGE': 4, 'AIR POLLUTION': 5, 'GARDENING': 6, 'BOATING AND SAILING': 7,
#              'BASKETBALL': 8, 'CREDIT CARD USE': 9, 'LATIN AMERICA': 10, 'FAMILY LIFE': 11, 'METRIC SYSTEM': 12,
#              'BASEBALL': 13, 'TAXES': 14, 'BOOKS AND LITERATURE': 15, 'CRIME': 16, 'PUBLIC EDUCATION': 17,
#              'RIGHT TO PRIVACY': 18, 'AUTO REPAIRS': 19, 'MIDDLE EAST': 20, 'FOOTBALL': 21,
#              'UNIVERSAL PBLIC SERV': 22, 'CAMPING': 23, 'FAMILY FINANCE': 24, 'POLITICS': 25, 'SOCIAL CHANGE': 26,
#              'DRUG TESTING': 27, 'COMPUTERS': 28, 'BUYING A CAR': 29, 'WOODWORKING': 30, 'EXERCISE AND FITNESS': 31,
#              'GOLF': 32, 'CAPITAL PUNISHMENT': 33, 'NEWS MEDIA': 34, 'HOME REPAIRS': 35, 'PAINTING': 36,
#              'FISHING': 37, 'SOVIET UNION': 38, 'CHILD CARE': 39, 'IMMIGRATION': 40, 'JOB BENEFITS': 41,
#              'RECYCLING': 42, 'MUSIC': 43, 'TV PROGRAMS': 44, 'ELECTIONS AND VOTING': 45, 'FEDERAL BUDGET': 46,
#              'MOVIES': 47, 'AIDS': 48, 'HOUSES': 49, 'VACATION SPOTS': 50, 'VIETNAM WAR': 51, 'CONSUMER GOODS': 52,
#              'RECIPES/FOOD/COOKING': 53, 'GUN CONTROL': 54, 'CLOTHING AND DRESS': 55, 'MAGAZINES': 56,
#              'SVGS & LOAN BAILOUT': 57, 'SPACE FLIGHT AND EXPLORATION': 58, "WOMEN'S ROLES": 59,
#              'PUERTO RICAN STTEHD': 60, 'TRIAL BY JURY': 61, 'ETHICS IN GOVERNMENT': 62, 'FAMILY REUNIONS': 63,
#              'RESTAURANTS': 64, 'UNIVERSAL HEALTH INS': 65}


### Dialogue act label encoding, MRDA
# {'S':0, 'B':1, 'D':2, 'F':3, 'Q':4}

### Dialogue act label encoding, DyDA
# {1:0, 2:1, 3:2, 4:3}

### Topic label encoding, DyDA
# {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}

class DialogueActData(Dataset):
    def __init__(self, corpus, phase, chunk_size=0, max_length=0):
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        df = pd.read_csv(f'data/{corpus}/{phase}.csv')
        max_conv_len = df['conv_id'].value_counts().max()
        if (chunk_size == 0 and phase == 'train') or phase != 'train':
            chunk_size = max_conv_len

        texts_all = df['text'].tolist()
        processed_texts = []
        for text in texts_all:
            text = text.strip().lower()
            text_tmp = ''
            pre_letter = ''
            for letter in text:
                if letter.isalpha() or letter.isdigit():
                    text_tmp = text_tmp + letter
                    pre_letter = letter
                elif letter in set(',.!?'):
                    if pre_letter != ' ':
                        text_tmp = text_tmp + ' ' + letter
                    else:
                        text_tmp = text_tmp + letter
                    pre_letter = letter
                else:
                    text_tmp = text_tmp + ' '
                    pre_letter = ' '
            processed_texts.append(text_tmp.strip())
        
        processed_texts = [re.sub(r'\s+', ' ', text) for text in processed_texts]

        if max_length == 0:
            encodings_all = tokenizer(texts_all, truncation=True, padding=True)
        else:
            encodings_all = tokenizer(texts_all, truncation=True, padding=True, max_length=max_length)
        input_ids_all = np.array(encodings_all['input_ids'])
        attention_mask_all = np.array(encodings_all['attention_mask'])

        input_ids_ = []
        attention_mask_ = []
        labels_ = []
        chunk_lens_ = []
        speaker_ids_ = []
        topic_labels_ = []
        chunk_attention_mask_ = []

        conv_ids = df['conv_id'].unique()
        for conv_id in tqdm(conv_ids, ncols=80, desc='processing data'):
            mask_conv = df['conv_id'] == conv_id
            df_conv = df[mask_conv]
            input_ids = input_ids_all[mask_conv]
            attention_mask = attention_mask_all[mask_conv]
            speaker_ids = df_conv['speaker'].values
            labels = df_conv['act'].values
            topic_labels = df_conv['topic'].values

            chunk_indices = list(range(0, df_conv.shape[0], chunk_size)) + [df_conv.shape[0]]  # 把一轮对话按chunk_size划分
            for i in range(len(chunk_indices) - 1):
                idx1, idx2 = chunk_indices[i], chunk_indices[i + 1]

                chunk_input_ids = input_ids[idx1: idx2].tolist()
                chunk_attention_mask = attention_mask[idx1: idx2].tolist()
                chunk_labels = labels[idx1: idx2].tolist()
                chunk_speaker_ids = speaker_ids[idx1: idx2].tolist()
                chunk_topic_labels = topic_labels[idx1: idx2].tolist()
                chunk_len = idx2 - idx1

                if idx2 - idx1 < chunk_size:  # pad成chunk_size
                    length1 = idx2 - idx1
                    length2 = chunk_size - length1
                    encodings_pad = [[0] * len(input_ids_all[0])] * length2
                    chunk_input_ids.extend(encodings_pad)
                    chunk_attention_mask.extend(encodings_pad)
                    labels_padding = np.array([-1] * length2)
                    chunk_labels = np.concatenate((chunk_labels, labels_padding), axis=0)
                    speaker_ids_padding = np.array([2] * length2)
                    chunk_speaker_ids = np.concatenate((chunk_speaker_ids, speaker_ids_padding), axis=0)
                    topic_labels_padding = np.array([99] * length2)
                    chunk_topic_labels = np.concatenate((chunk_topic_labels, topic_labels_padding), axis=0)

                input_ids_.append(chunk_input_ids)
                attention_mask_.append(chunk_attention_mask)
                labels_.append(chunk_labels)
                chunk_lens_.append(chunk_len)
                speaker_ids_.append(chunk_speaker_ids)
                topic_labels_.append(chunk_topic_labels)
                chunk_attention_mask_tmp = [1] * chunk_len + [0] * (chunk_size - chunk_len)
                chunk_attention_mask_.append(chunk_attention_mask_tmp)


        # print('Done')

        self.input_ids = input_ids_
        self.attention_mask = attention_mask_
        self.labels = labels_
        self.chunk_lens = chunk_lens_
        self.speaker_ids = speaker_ids_
        self.topic_labels = topic_labels_
        self.chunk_attention_mask = chunk_attention_mask_

    def __getitem__(self, index):
        item = {
            'input_ids': torch.tensor(self.input_ids[index]),  # [B, chunk_size, max_len]
            'attention_mask': torch.tensor(self.attention_mask[index]),  # [B, chunk_size, max_len]
            'labels': torch.tensor(self.labels[index]),  # [B, chunk_size]
            'chunk_lens': torch.tensor(self.chunk_lens[index]),  # [B]
            'speaker_ids': torch.tensor(self.speaker_ids[index], dtype=torch.long),  # [B, chunk_size]
            'topic_labels': torch.tensor(self.topic_labels[index], dtype=torch.long),  # [B, chunk_size]
            'chunk_attention_mask': torch.tensor(self.chunk_attention_mask[index], dtype=torch.long)  # [B, chunk_size]
        }
        return item

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):  # batch是字典的列表
    output = {}
    output['input_ids'] = torch.stack([x['input_ids'] for x in batch], dim=0)  # [B, chunk_size, max_len]
    output['attention_mask'] = torch.stack([x['attention_mask'] for x in batch], dim=0)  # [B, chunk_size, max_len]
    output['labels'] = torch.stack([x['labels'] for x in batch], dim=0)  # [B, chunk_size]
    output['chunk_lens'] = torch.stack([x['chunk_lens'] for x in batch], dim=0)  # [B]
    output['speaker_ids'] = torch.stack([x['speaker_ids'] for x in batch], dim=0)  # [B, chunk_size]
    output['topic_labels'] = torch.stack([x['topic_labels'] for x in batch], dim=0)  # [B, chunk_size]
    output['chunk_attention_mask'] = torch.stack([x['chunk_attention_mask'] for x in batch], dim=0)  # [B, chunk_size]
    return output



def data_loader(corpus, phase, batch_size, chunk_size=0, shuffle=False, max_length=0):
    dataset = DialogueActData(corpus, phase, chunk_size=chunk_size, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


if __name__ == '__main__':
    dataset = DialogueActData(corpus='swda', phase='train', chunk_size=128)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch in dataloader:
        print(batch['input_ids'].shape)
        print(batch['labels'].shape)
        print(batch['chunk_lens'].shape)
        print(batch['speaker_ids'].shape)
        print(batch['topic_labels'].shape)
        break
