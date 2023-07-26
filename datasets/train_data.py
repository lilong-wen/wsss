import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import csv
import pickle
import torch
import random
from PIL import Image
from dataclasses import dataclass
import utils.misc as utils


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms,  char_dict_pth, sep="\t", 
                 single_text=False, text_batch_size=256, vocab_size=40000, context_length=77, image_resolution=512):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep, quoting=csv.QUOTE_NONE)

        img_key = 'filepath'
        caption_key = 'title'
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.single_text = single_text
        self.vocab_size = vocab_size
        self.text_batch_size = text_batch_size
        with open(char_dict_pth, 'rb') as f:
            self.letters = pickle.load(f)
            self.letters = [chr(x) for x in self.letters]
        self.p2idx = {p: idx+1 for idx, p in enumerate(self.letters)}
        self.idx2p = {idx+1: p for idx, p in enumerate(self.letters)}

        self.idx_mask = len(self.letters) + 1
        self.EOS = len(self.letters) + 2
        self.image_resolution = image_resolution

        self.max_len = 32
        self.word_len = 25

        self.context_length = context_length

        logging.debug('Done loading data.')

    def tokenize(self, text):
        token = torch.zeros(self.word_len)
        for i in range(min(len(text), self.word_len)):
            token[i] = self.p2idx[text[i]]
        if len(text) >= self.word_len:
            token[-1] = self.EOS
        else:
            token[len(text)] = self.EOS

        return token

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        ori_images = Image.open(str(self.images[idx]))

        images = self.transforms(ori_images)

        all_texts = self.captions[idx].split(' ')
        texts = torch.zeros((self.max_len, self.word_len))
        texts_f = torch.zeros((self.max_len, self.word_len))
        masked_chars = torch.zeros(self.max_len)
        for i in range(min(len(all_texts), self.max_len)):
            t = self.tokenize(all_texts[i])
            texts_f[i] += t
            rand_idx = random.randint(0, min(len(all_texts[i]), self.word_len) - 1)
            masked_chars[i] = t[rand_idx].clone()
            t[rand_idx] = self.idx_mask
            texts[i] += t

        # image masks can be used to mask out the padding regions during training
        image_masks = torch.zeros((self.image_resolution // 32, self.image_resolution // 32), dtype=torch.bool)

        # return images, texts.long(), masked_chars.long(), image_masks
        #TODO change the second text to unmasked text
        return images, texts_f.long(), {"texts_pad": texts.long(), "chars": masked_chars.long(), \
                                        'ori_size': torch.tensor(ori_images.size),
                                        'size': torch.tensor(images.shape[1:])}
       
@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def get_csv_dataset(args, preprocess_fn):
    
    input_filenames = args.train_file
    datasets_list = []
    for filename in input_filenames:
        dataset_item = CsvDataset(filename,
                            preprocess_fn,
                            char_dict_pth=args.char_dict_path
                            )
        datasets_list.append(dataset_item)

    num_samples = sum([len(dataset) for dataset in datasets_list])

    if len(datasets_list) > 1:
        dataset = ConcatDataset(datasets_list) 
    else:
        dataset = datasets_list[0]

    assert num_samples == len(dataset)
    
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = sampler is None 

    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=shuffle,
                                            collate_fn=utils.collate_fn,
                                            num_workers=args.num_workers,
                                            pin_memory=True,
                                            sampler=sampler,
                                            drop_last=True)

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    tmp_sample = next(iter(dataloader))
    aa = tmp_sample[0].tensors

    return DataInfo(dataloader, sampler)
            

def get_train_data(args, preprocess_fns):

    preprocess_fn = preprocess_fns

    data = get_csv_dataset(args, preprocess_fn)
    
    return data