import os

train_load_dir = './decoder_model/decoder_model_info'
train_decoder_model = './original_pretrained_model_for_bertGPT.pth'
train_data = "./data_pth/train_data_info.pth"
val_data = "./data_pth/train_data_info.pth"
test_data = "./data_pth/test_data_info.pth"
trained_decoder_model = os.path.join(train_load_dir, "8decoder.pth")
gen_file_path = './generate/generate_sentences_info.txt'
