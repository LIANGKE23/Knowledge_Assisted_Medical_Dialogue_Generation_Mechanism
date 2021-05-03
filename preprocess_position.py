from pytorch_pretrained_bert import BertTokenizer
import torch
import os
import json
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

MAX_ENCODER_SIZE = 400
MAX_DECODER_SIZE = 100


def seq2token_ids(source_seqs, target_seq, source_seqs_info):
    # 可以尝试对source_seq进行切分
    encoder_input = []
    encoder_position_id = []
    for index, source_seq in enumerate(source_seqs):
        # 去掉 xx：
        encoder_input_tmp = tokenizer.tokenize(source_seq[3:])+ ["[SEP]"]
        encoder_position_id += [index + 1] * (len(encoder_input_tmp))
        encoder_input += encoder_input_tmp
    decoder_input = ["[CLS]"] + tokenizer.tokenize(target_seq[3:])  # 去掉 xx：
    encoder_input_info = tokenizer.tokenize(source_seqs_info)
    if len(encoder_input_info) > 200:
        encoder_input_info = encoder_input_info[:200] + ["[SEP]"]
    # 设置不得超过 MAX_ENCODER_SIZE 大小
    if len(encoder_input) > MAX_ENCODER_SIZE - 1 - len(encoder_input_info):
        if "[SEP]" in encoder_input[-(MAX_ENCODER_SIZE - len(encoder_input_info)):-1]:
            idx = encoder_input[:-1].index("[SEP]", -((MAX_ENCODER_SIZE - len(encoder_input_info)) - 1))
            encoder_input = encoder_input[idx + 1:]
            encoder_position_id = encoder_position_id[idx + 1:]

    encoder_input = ["[CLS]"] + encoder_input_info + encoder_input[-((MAX_ENCODER_SIZE - len(encoder_input_info)) - 1):]
    encoder_position_id = [0]*(len(encoder_input_info)+1)+encoder_position_id[-((MAX_ENCODER_SIZE - len(encoder_input_info)) - 1):]
    decoder_input = decoder_input[:MAX_DECODER_SIZE - 1] + ["[SEP]"]
    enc_len = len(encoder_input)
    pos_len = len(encoder_position_id)
    print(max(encoder_position_id))
    dec_len = len(decoder_input)
    assert enc_len == pos_len
    assert enc_len <=400
    
    # conver to ids
    encoder_input = tokenizer.convert_tokens_to_ids(encoder_input)
    decoder_input = tokenizer.convert_tokens_to_ids(decoder_input)

    return encoder_input, decoder_input, encoder_position_id


def make_dataset(data, file_name='train_data.pth'):
    train_data = []

    for d in tqdm(data):
        d0 = d[0]
        d = d[1:]
        d_len = len(d)
        for i in range(d_len // 2):
            encoder_input, decoder_input, encoder_position_id = seq2token_ids(d[:2 * i + 1], d[2 * i + 1], d0)
            train_data.append((encoder_input, encoder_position_id, decoder_input))

    encoder_input, encoder_position_id,\
    decoder_input = zip(*train_data)

    encoder_input = list(encoder_input)
    encoder_position_id = list(encoder_position_id)
    decoder_input = list(decoder_input)

    encoder_len = len(encoder_input)
    encoder_position_id_len = len(encoder_position_id)
    decoder_len = len(decoder_input)

    train_data = [encoder_input, encoder_position_id, decoder_input]
    # torch.save(train_data, file_name)


def process_dataset(dataset_file='train_data.json', data_file='train_data.pth'):
    with open(dataset_file, "r", encoding='utf-8') as f:
        json_data = f.read()
        data = json.loads(json_data)
    make_dataset(data, data_file)
    pass



print(f'Process the train dataset\n')
process_dataset('./data/data_json/train_data_new.json', './data_pth/train_data_position.pth')

print(f'Process the test dataset\n')
process_dataset("./data/data_json/validate_data_new.json", "./data_pth/validate_data_position.pth")

print(f'Process the test dataset\n')
process_dataset('./data/data_json/test_data_new.json', './data_pth/test_data_position.pth')