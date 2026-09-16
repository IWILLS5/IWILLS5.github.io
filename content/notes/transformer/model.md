# 以下代码参考如下：[从零实现 Transformer](https://songyaolun.github.io/transformer-from-scratch/)


```python
import torch 
import torch.nn as nn 
from torch.utils import data
import math
```

# 小词库


```python
raw_data = [
    ("I love deep learning", "我爱深度学习"),
    ("Transformer is all your need", "你需要的就是变形金刚"),
    ("Attention mechanism is great", "注意力机制很棒"),
    ("hello world", "你好世界"),
    ("PyTorch is easy to learn", "PyTorch很容易学")
]

#构建词表
#src是所有输入的token的集合
src_vocab = {'<pad>' : 0, '<sos>' : 1, '<eos>' : 2}
#tgt是所有输出的token的集合
tgt_vocab = {'<pad>' : 0, '<sos>' : 1, '<eos>' : 2}
 
#遍历raw_data，填充src和tgt词表
for src, tar in raw_data:
    for word in src.split():
        if word not in src_vocab:
            src_vocab[word] = len(src_vocab)
    for word in tar:
        if word not in tgt_vocab:
            tgt_vocab[word] = len(tgt_vocab)

#构建由id序号得到token（src和tgt）
idx2src = {v : k for k, v in src_vocab.items()}
idx2tar = {v : k for k, v in tgt_vocab.items()}
```

# 定义数据集


```python
class TranslationDataSet(data.Dataset):
    #传入数据， 特征数据对应的id， 标签数据对应的id
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    # 数据的长度
    def __len__(self):
        return len(self.data)
    # 获取一个训练数据（id表示的样子）
    def __getitem__(self, index):
        src_text, tgt_text = self.data[index]

        src_indices = [self.src_vocab[word] for word in src_text.split()]
        tgt_indices = [self.tgt_vocab[word] for word in tgt_text]
        return torch.tensor(src_indices), torch.tensor(tgt_indices)
```

# DataLoader 和 自定义collate_fn


```python
SOS_TENSOR = torch.tensor([tgt_vocab['<sos>']])
EOS_TENSOR = torch.tensor([tgt_vocab['<eos>']])
#自定义collate_fn函数，补充SOS、EOS、PAD标识
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_data, tgt_data in batch:
        src_batch.append(src_data)
        tgt_process = torch.cat([
            SOS_TENSOR,
            tgt_data,
            EOS_TENSOR
        ])
        tgt_batch.append(tgt_process)
    # 自动选取最长token的维度为统一维度，并补充padding以及stack堆叠新的一层
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    return src_batch, tgt_batch
```


```python
data_set = TranslationDataSet(raw_data, src_vocab, tgt_vocab)
data_set[0]
train_loader = data.DataLoader(data_set, 2, shuffle=False, collate_fn=collate_fn)
```

# 词嵌入


```python
emb_layer = nn.Embedding(num_embeddings=10, embedding_dim=4)
print(f'Embedding 权重矩阵形状：{emb_layer.weight.shape}')
# print(f'{emb_layer.weight = }')
input_ids = torch.tensor([[3, 0, 7], [3, 0, 7]])
outputs =emb_layer(input_ids)

print(f'输出形状:{outputs.shape}')
# print(f'{outputs = }')

print(torch.equal(emb_layer.weight.data[3], outputs.data[0]))
```

    Embedding 权重矩阵形状：torch.Size([10, 4])
    输出形状:torch.Size([2, 3, 4])
    False
    


```python
d_model = 512
emb_large = nn.Embedding(100, d_model)
sample = emb_large(torch.tensor([1]))

print(f'未缩放的标准差：{sample.std().item():.4f}')
sample_scaled = sample * math.sqrt(d_model)
print(f'缩放后的标准差：{sample_scaled.std().item():.4f}')
print(f"缩放因子 sqrt({d_model}) = {math.sqrt(d_model):.2f}")
```

    未缩放的标准差：1.0069
    缩放后的标准差：22.7844
    缩放因子 sqrt(512) = 22.63
    

# 位置编码

$PE(pos, 2i) = sin(pos / 10000^{2i/d_{model}})$<br>
$PE(pos, 2i + 1) = cos(pos / 10000^{2i/d_{model}})$


```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #所有行的奇数列
        pe[:, 1::2] = torch.cos(position * div_term)
        #所有行的偶数列
        pe[:, 0::2] = torch.sin(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    # x的shape是 [batch, ses_len, d_model]
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
```


```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(self.d_model)
class TransformerInputLayer(nn.Module):
    # 论文中提到，dropout在sub_layer的输出之后，before it is added to the sub_layer input and normalized
    def __init__(self, vocab_size, d_model, max_len = 5000, dropout = 0.1):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positon_code = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.embedding(x) + self.positon_code(x))

```
