from tqdm.auto import tqdm
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM, AdamW
import torch
import os

VOCAB_SIZE = 30_522
# 读取无标签数据
def read_unlabeled_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    X = [line.strip() for line in lines]
    return X


# 分割无标签数据
def split_unlabeled_data(file_path, chunk_size=10_000):
    X = read_unlabeled_data(file_path)
    file_count = 0
    if not os.path.exists('data/process'):
        os.makedirs('data/process')
    for i in tqdm(range(0, len(X), chunk_size)):
        with open(f'data/process/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(X[i:i+chunk_size]))
        file_count += 1

def build_tokenizer():
    paths = [str(x) for x in Path('data/process').glob('*.txt')]
    #print(paths)
    path = 'data/raw/nolabel.txt'
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=VOCAB_SIZE, min_frequency=5,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
    os.makedirs('data/tokenizer', exist_ok=True)
    tokenizer.save_model('data/tokenizer')
    return tokenizer

def mlm(tensor):
    rand = torch.rand(tensor.shape)
    mask_arr = (rand < .15) * (tensor != 0) * (tensor != 1) * (tensor != 2)
    for i in range(tensor.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        tensor[i, selection] = 4
        return tensor
    
def load_data():
    paths = [str(x) for x in Path('data/process').glob('*.txt')]
    #paths = ['data/raw/nolabel.txt']
    tokenizer = RobertaTokenizer.from_pretrained('data/tokenizer', max_len=512)
    input_ids = []
    mask = []
    labels = []
    for path in tqdm(paths):
        with open (path, 'r', encoding='utf-8') as fp:
            lines = fp.read().split('\n')
        sample = tokenizer(lines, max_length=512, padding='max_length', truncation=True,return_tensors='pt')
        labels.append(sample.input_ids)
        mask.append(sample.attention_mask)
        input_ids.append(mlm(sample.input_ids.detach().clone()))
    labels = torch.cat(labels)
    mask = torch.cat(mask)
    input_ids = torch.cat(input_ids)
    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return self.encodings['input_ids'].shape[0]
        def __getitem__(self, i):
            return {key: tensor[i] for key, tensor in self.encodings.items()}
    dataset = Dataset(encodings)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader
    
def build_model():
    config = RobertaConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=514,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,#注意这个地方与标准的不一致
        type_vocab_size=1
    )
    model = RobertaForMaskedLM(config)
    return model

def train_model(dataloader, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    optim = AdamW(model.parameters(), lr=5e-5)
    epochs = 3
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for index, batch in enumerate(loop): 
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            loop.set_description(f'Loss: {loss.item()}')
            # 增加checkpoint保存点
            if index % 10000 == 0 and index > 0:
                torch.save(model.state_dict(), f'data/pretrain/checkpoint_{epoch}_{index}.pt')
        # 每个epoch保存一次模型
        model.save_pretrained('data/pretrain')

#split_unlabeled_data('data/raw/nolabel.txt')
build_tokenizer()
dataload = load_data()
model = build_model()
train_model(dataload, model)