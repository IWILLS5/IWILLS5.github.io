# 以下代码参考如下：[从零实现 Transformer](https://songyaolun.github.io/transformer-from-scratch/)，原本实现有错误的地方，在实现时会有修改说明


```python
import torch 
import torch.optim as optim
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
print(f'{outputs = }')

print(torch.equal(emb_layer.weight.data[3], outputs.data[0]))
```

    Embedding 权重矩阵形状：torch.Size([10, 4])
    输出形状:torch.Size([2, 3, 4])
    outputs = tensor([[[-0.8530, -1.2476,  0.0060,  0.5094],
             [-0.2687, -0.6593, -1.3886, -0.7098],
             [ 0.2556,  1.2938, -1.1191, -0.1350]],
    
            [[-0.8530, -1.2476,  0.0060,  0.5094],
             [-0.2687, -0.6593, -1.3886, -0.7098],
             [ 0.2556,  1.2938, -1.1191, -0.1350]]], grad_fn=<EmbeddingBackward0>)
    False
    


```python
d_model = 512

std_init = 1.0 / math.sqrt(d_model)  # 1 / sqrt(512) ≈ 0.04419
embedding = nn.Embedding(100, d_model)
nn.init.normal_(embedding.weight, mean=0.0, std=std_init)

sample = embedding(torch.tensor([1]))

print(f'未缩放的标准差：{sample.std().item():.4f}')
sample_scaled = sample * math.sqrt(d_model)
print(f'缩放后的标准差：{sample_scaled.std().item():.4f}')
print(f"缩放因子 sqrt({d_model}) = {math.sqrt(d_model):.2f}")
```

    未缩放的标准差：0.0446
    缩放后的标准差：1.0093
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
        x = x + self.pe[ : , : x.size(1), : ]
        return x
```


```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        #此处教程中有错误， nn.embedding的mean为0，std为1， 我们需要手动设置标准差为 1.0 / math.sqrt(d_model)，方差为1.0 / d_model
        # 之所以设置为方差为1.0 / d_model，输入端的词嵌入矩阵与解码器最末端的线性分类头（nn.Linear(d_model, vocab_size)）共用了同一套参数矩阵 W，
        # 最后会做点积，为了防止点积的结果爆炸，也是防止后面的softmax概率变为极端尖刺
        
        std_init = 1.0 / math.sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=std_init)
        self.d_model = d_model

    def forward(self, x):
        # 原论文规范：若 Embedding 权重与输出层共享（初始化 std=1/sqrt(d_model)），
        # 需乘以 sqrt(d_model) 将词向量尺度放大回 1，以匹配位置编码（[-1, 1]）的数值量级。
        return self.embedding(x) * math.sqrt(self.d_model)


        
class TransformerInputLayer(nn.Module):
    # 论文中提到，dropout在sub_layer的输出之后，before it is added to the sub_layer input and normalized
    def __init__(self, vocab_size, d_model, max_len = 5000, dropout = 0.1):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positon_code = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = self.embedding(x)
        out = self.positon_code(out)
        return self.dropout(out)

```

# 注意力机制的实现


```python
class MutiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads        
        self.d_model = d_model
        self.n_heads = num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, masked = None):
        #q,k,v进入的形状是（batch， seq_len, d_model）
        batch = q.size(0)
        #维度由（batch， seq_len, d_model）-> （batch，seq_len, heads, d_model // heads） -> （batch， heads, seq_len, d_model // heads）
        #不直接写成self.W_Q(q).view(batct, self.n_heads, -1, self.d_k),是因为这样子会打乱数据的顺序
        #因为一开始数据是在内存中连续分布，假如token_0的embedding vector是 x1, x2, x3, x4,token_1的是y1, y2, y3, y4,heads = 2
        #内存中会先放token_0的embedding vector，然后是token_1,使用self.W_Q(q).view(batct, -1, self.n_heads, self.d_k)
        #可以保证第一个head关于token_0的内容（Q[0,0,0,0]和Q[0,0,0,1]）就是x1,x2;第二个head关于token_0的内容
        #  (Q[0,0,1,0]和Q[0,0,1,1])是x3,x4,即每个head得到正确的数据
        #如果直接使用self.W_Q(q).view(batct, self.n_heads, -1, self.d_k)
        #此时Q[0,0,0,0]和Q[0,0,0,1]仍然是x1,x2，但它的含义是第一个head中token_0的内容，很明显这个部分是对的;
        # 但Q[0,0,1,0]和Q[0,0,1,1]虽然仍是x3,x4，但它的含义是第一个head中token_1的内容，应该是y1, y2,很明显错了
        Q = self.W_Q(q).view(batch, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.W_K(k).view(batch, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.W_V(v).view(batch, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # Q * K / sqrt(d_model // heads)  shape 为（batch，heads，seq_len, seq_len）
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        #decode中用于避免标签泄露 或者 交叉注意力去除pad
        if masked is not None:
            scores = scores.masked_fill(masked == 0, -1e9)
        #softmax计算得分
        attention = self.attn_dropout(nn.functional.softmax(scores, dim=-1))
        #加权求和， 形状变为（batch, heads, seq_len, d_model // heads）
        context = torch.matmul(attention, V)
        #将heads拼接在一起，先内存连续化，然后使用view()
        context = context.permute(0, 2, 1, 3).contiguous().view(batch, -1, self.d_model)
    
        return self.W_O(context), attention

```

# FFN


```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # 激活函数用来增加非线性
        return self.fc2(self.dropout(nn.functional.relu(self.fc1(x))))

```

## 按理说，在MutiHeadAttention后要加dropout，FNN第二层的fc2也要加dropout，但我们设计在Add模块内，因为两个操作后面都是残差连接


```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.mha = MutiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    # 主流实现是pre—norm，即x = x + SubLayer(Norm(x))， 论文实现的是x = Norm(x + SubLayer(x)),此处实现pre_norm,因为后面作者在组装的时候最后加了
    # 一个norm，所以不按照原文来，而是使用pre_norm
    def forward(self, x, mask):
        norm_x = self.norm1(x)
        mha_op, _ = self.mha(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(mha_op)

        norm_x = self.norm2(x)
        ffn_op = self.ffn(norm_x)
        x = x + self.dropout(ffn_op)
        return x
```


```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.mha1 = MutiHeadAttention(d_model, num_heads) #自注意力
        self.mha2 = MutiHeadAttention(d_model, num_heads) #交叉注意力
        self.ffn = FFN(d_model, d_ff) #ffn
        self.norm1 = nn.LayerNorm(d_model) #自注意力之后
        self.norm2 = nn.LayerNorm(d_model) #交叉注意力之后
        self.norm3 = nn.LayerNorm(d_model) #ffn之后

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_op, src_mask, tgt_mask):
        norm_x = self.norm1(x)
        #标签 自注意力，   掩码用于预防标签泄露
        mha_op, _ = self.mha1(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout(mha_op)

        norm_x = self.norm2(x)
        # decoder的Q和encoder的K，V 进行交叉注意力， 不对其使用防止标签泄露的掩码，
        # 是因为encoder中的内容属于已知内容，只需要将encoder中的pad填充掩盖即可
        cro_mha_op, attn_map = self.mha2(norm_x, encoder_op, encoder_op, src_mask)
        
        x = x + self.dropout(cro_mha_op)

        norm_x = self.norm3(x)
        # 进行两层全连接层，增加非线性
        ffn_op = self.ffn(norm_x)
        x = x + self.dropout(ffn_op)
        return x, attn_map
```

- 到了这一步，该实现decoder的拼装，了解到在decoder中需要使用masked MultiHeadAttention来防止标签泄露，在交叉注意力的地方也要使用padding mask来掩盖在encoder中输出的内容的pad，以避免pad对softmax的干扰，所以在encoder中也需要添加mask


```python
def make_pad_mask(seq, pad_idx = 0):
    # 这里传入的seq就是dateloader输出的数据（batch， seq_len），这里的seq可以是src_len或者tgt_len,如果是在encoder中使用padding mask就是src_len
    # 之所以讲它生维，是为了对注意力得到的score进行掩码，这样子mask就会自动对其增加的heads和seq_len层（这里的seq如果是自注意力，和上方的可能一样
    # 如果是交叉注意力，那么就是src_len）进行广播
    return seq.ne(pad_idx).unsqueeze(1).unsqueeze(2)
```


```python
def make_subsequent_mask(seq_len):

    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8))
    return mask.unsqueeze(0).unsqueeze(0)
```

- Q: 为什么交叉注意力的mask是在batch和sql_len中添加两个维度， 而因果掩码是在前两个维度?
- A: 首先交叉注意力的mask因为每个句子的padding填充不一样，所以对于仅仅是对于这个batch下的scores进行掩码；但是因果掩码对于所有batch的decoder的输入都是在最后两个维度进行下三角掩码，所以仅仅一个mask矩阵直接广播所有batch和heads就好


```python
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, n_layer, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(n_layer)
        )
        self.norm = nn.LayerNorm(d_model)
    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)
```


```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, n_layers, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(n_layers)
        )

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x, encoder_op, src_mask, tgt_mask):
        for layer in self.layers:
            tgt, attn_map = layer(x, encoder_op, src_mask, tgt_mask)
        tgt = self.norm(tgt)
        return self.fc(tgt), attn_map
```


```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.src_embedding = TransformerInputLayer(src_vocab_size, d_model, dropout=dropout)
        self.tgt_embedding = TransformerInputLayer(tgt_vocab_size, d_model, dropout=dropout)

        self._encoder = Encoder(d_model, num_heads, d_ff, n_layers, dropout)
        self._decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, n_layers, dropout)

        self.src_pad_idx = 0
        self.tgt_pad_idx = 0

    def make_masks(self, src, tgt):
        src_mask = make_pad_mask(src, self.src_pad_idx)
        tgt_mask = make_pad_mask(tgt, self.tgt_pad_idx)

        subsequent_mask = make_subsequent_mask(tgt.size(1)).to(src.device)
        combined_mask = tgt_mask & subsequent_mask
        return src_mask, combined_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.make_masks(src, tgt)
        enc_op = self._encoder(self.src_embedding(src), src_mask)
        dec_op, _ = self._decoder(self.tgt_embedding(tgt), enc_op, src_mask, tgt_mask)
        return dec_op
    
    def encode(self, src, src_mask):
        return self._encoder(self.src_embedding(src), src_mask)

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        dec_output, _ = self._decoder(self.tgt_embedding(tgt), enc_output, src_mask, tgt_mask)
        return dec_output
```

- Q:为什么decoder对于标签的一次输入 相当于是集合时间步的并行?
- A: decoder进行的第一层masked multiheadsattention得到的scores进行mask后，得到的内容就类似seq_len次并行计算


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
src_vocab_size = 100
tgt_vocab_size = 100
d_model = 512
n_layers = 2
n_head = 8

model = Transformer(len(src_vocab), len(tgt_vocab), d_model, n_head, d_ff=2048, n_layers=n_layers)
model.to(device)

#损失函数 
criterion = nn.CrossEntropyLoss(ignore_index=0)
#优化器
optimizer = optim.Adam(model.parameters(), lr = 0.0001, betas=(0.9, 0.98), eps=1e-9)
model.train()

epochs = 100
for epoch in range(epochs):
    epoch_loss = 0.0

    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_i, tgt_o = tgt[:, :-1], tgt[:, 1:]
        #清空梯度
        optimizer.zero_grad()

        output = model(src, tgt_i)
        #损失
        loss = criterion(
            output.contiguous().view(-1, len(tgt_vocab)),
            tgt_i.contiguous().view(-1)
        )
        #反向传播
        loss.backward()
        optimizer.step()

        epoch_loss += loss
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}')
print("训练完成！")
```

    cuda
    Epoch [10/100], Loss: 0.7032
    Epoch [20/100], Loss: 0.0785
    Epoch [30/100], Loss: 0.0309
    Epoch [40/100], Loss: 0.0204
    Epoch [50/100], Loss: 0.0122
    Epoch [60/100], Loss: 0.0084
    Epoch [70/100], Loss: 0.0064
    Epoch [80/100], Loss: 0.0043
    Epoch [90/100], Loss: 0.0031
    Epoch [100/100], Loss: 0.0023
    训练完成！
    

### Test code


```python
d_model = 512
num_heads = 8
d_ff = 2048
batch_size = 2
src_len = 10
tgt_len = 5

# 测试 Encoder Layer
enc_layer = EncoderLayer(d_model, num_heads, d_ff)
src_input = torch.randn(batch_size, src_len, d_model)
enc_output = enc_layer(src_input, mask=None)
print(f"Encoder Input:  {src_input.shape}")   # (2, 10, 512)
print(f"Encoder Output: {enc_output.shape}")   # (2, 10, 512)

# 测试 Decoder Layer
dec_layer = DecoderLayer(d_model, num_heads, d_ff)
tgt_input = torch.randn(batch_size, tgt_len, d_model)
dec_output, cross_attn = dec_layer(tgt_input, enc_output, src_mask=None, tgt_mask=None)
print(f"Decoder Input:  {tgt_input.shape}")    # (2, 5, 512)
print(f"Decoder Output: {dec_output.shape}")    # (2, 5, 512)
print(f"Cross Attention: {cross_attn.shape}")   # (2, 8, 5, 10)
```

    Encoder Input:  torch.Size([2, 10, 512])
    Encoder Output: torch.Size([2, 10, 512])
    Decoder Input:  torch.Size([2, 5, 512])
    Decoder Output: torch.Size([2, 5, 512])
    Cross Attention: torch.Size([2, 8, 5, 10])
    


```python
a = torch.tensor([[1, 2, 3], [3, 4, 5]])
a.view(-1, 1)
```




    tensor([[1],
            [2],
            [3],
            [3],
            [4],
            [5]])




```python
a = torch.rand((4, 4))
print(f'{a = }')
b = torch.tril(a)
print(f'{b = }')
```

    a = tensor([[0.4364, 0.1383, 0.5314, 0.1945],
            [0.8330, 0.9127, 0.5264, 0.0318],
            [0.9959, 0.5765, 0.7143, 0.2450],
            [0.0412, 0.9190, 0.8502, 0.6174]])
    b = tensor([[0.4364, 0.0000, 0.0000, 0.0000],
            [0.8330, 0.9127, 0.0000, 0.0000],
            [0.9959, 0.5765, 0.7143, 0.0000],
            [0.0412, 0.9190, 0.8502, 0.6174]])
    


```python
seq = torch.tensor([[1, 2, 0, 0]])  # 0 是 pad

pad_mask = make_pad_mask(seq, pad_idx=0)
print("Pad Mask:", pad_mask.squeeze())
# tensor([True, True, False, False])

subsequent_mask = make_subsequent_mask(5)
print("Subsequent Mask:\n", subsequent_mask.squeeze())
# tensor([[1, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0],
#         [1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1]])
```

    Pad Mask: tensor([ True,  True, False, False])
    Subsequent Mask:
     tensor([[1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]], dtype=torch.uint8)
    


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
src_vocab_size = 100
tgt_vocab_size = 100
d_model = 512
n_layers = 2
n_head = 8

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff=2048, n_layers=n_layers)
model.to(device)

# 伪造数据：Batch=2, Src_Len=5, Tgt_Len=6
# 0 是 pad，1 是 sos，2 是 eos
src = torch.tensor([[1, 5, 6, 2, 0], [1, 9, 2, 0, 0]]).to(device)
tgt = torch.tensor([[1, 7, 3, 4, 8, 2], [1, 6, 8, 2, 0, 0]]).to(device)

print("Source Shape:", src.shape)  # (2, 5)
print("Target Shape:", tgt.shape)  # (2, 6)

output = model(src, tgt)
print("Output Shape:", output.shape)  # (2, 6, 100) → (batch, tgt_len, vocab_size)
```

    cuda
    Source Shape: torch.Size([2, 5])
    Target Shape: torch.Size([2, 6])
    Output Shape: torch.Size([2, 6, 100])
    
