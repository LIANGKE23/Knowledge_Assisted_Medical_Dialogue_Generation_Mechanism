import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, TensorDataset, DataLoader

import fire

# uses allennlp modules
from allennlp.nn import util

# imports chinese gpt
from chinese_gpt import TransformerEncoder, TransformerDecoderLM
import conf

class MyDataset(Dataset):
    def __init__(self, *data):
        self.data = data

    def __getitem__(self, index):
        return tuple(data[index] for data in self.data)

    def __len__(self):
        return len(self.data[0])


def collate_fn(batch):
    pad_id = 0
    input_ids = []
    output_ids = []
    input_mask = []
    output_mask =[]

    btc_size = len(batch)
    max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    max_output_len = 0

    # 计算该batch中input的最大长度
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx][0]):
            max_input_len = len(batch[btc_idx][0])
        if max_output_len < len(batch[btc_idx][1]):
            max_output_len = len(batch[btc_idx][1])
    # 使用pad_id对小于max_input_len的input_id进行补全

    for btc_idx in range(btc_size):
        input_len = len(batch[btc_idx][0])
        input_ids.append(batch[btc_idx][0])
        input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))

        output_len = len(batch[btc_idx][1])
        output_ids.append(batch[btc_idx][1])
        output_ids[btc_idx].extend([pad_id] * (max_output_len - output_len))

        input_mask.append([1] * input_len + [pad_id] * (max_input_len - input_len))
        output_mask.append([1] * output_len + [pad_id] * (max_output_len - output_len))
    return tuple((torch.tensor(input_ids, dtype=torch.long), torch.tensor(output_ids, dtype=torch.long), torch.tensor(input_mask, dtype=torch.long), torch.tensor(output_mask, dtype=torch.long)))


class BertGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoderLM()

    def forward(self, encoder_input, mask_encoder_input, decoder_input, mask_decoder_input):
        _, past = self.encoder(encoder_input, mask_encoder_input)

        mask = torch.cat([mask_encoder_input, mask_decoder_input], dim=1)
        logits, _ = self.decoder(decoder_input, mask, past=past, past_length=0)

        return logits


def calculate_perplexity(
    batch_size=1,
    gpu_id=0,
    decoder_path=conf.trained_decoder_model
    ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    model = BertGPT()
    model.load_state_dict(torch.load(decoder_path))
    print(f'load from {decoder_path}')
    model = model.to(device)
    model.eval()
    print('load success')
    #------------------------END LOAD MODEL--------------

    test_data = torch.load(conf.test_data)
    test_dataset = MyDataset(*test_data)

    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=2, collate_fn=collate_fn)
    #------------------------END LOAD VAL DATA--------------

    perplexity = 0
    batch_count = 0
    print('start calculate the test perplexity....')

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch

            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)
  
            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
            perplexity += np.exp(loss.item())
            batch_count += 1


    print(f'test perplexity: {perplexity / batch_count}')


    #------------------------END VAL-------------------


if __name__ == '__main__':
    fire.Fire(calculate_perplexity)

