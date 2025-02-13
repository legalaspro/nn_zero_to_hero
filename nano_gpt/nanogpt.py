import torch 
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 64 # how many independent sequences to train at once in parallel
block_size = 256  # what is maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
if torch.cuda.is_available(): 
    device = torch.device("cuda") # A100 ~ 2.75x faster then ( ~ 11.8 min with 10mln parameters)
elif torch.backends.mps.is_available(): 
    device = torch.device("mps") #  ~ 8.35x faster than CPU (~ 32.5 min with 10mln parameters)
else:
    device = torch.device("cpu")
eval_iters = 200
n_embd = 384
n_head = 6 # 384/6 = 64
n_layer = 6
dropout = 0.2
#---------------------------------------------------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# here all the unique characters in the file
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a dictionary mapping characters to integers
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)} 
encode = lambda s: [stoi[c] for c in s] # encoder: Take in string, return list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: Take in list of integers, return string

# Train and Test splits
data = torch.tensor(encode(text), dtype=torch.int64)
n = int(0.9 * len(data)) # first 90% of the data will be training data, rest will be validation data
train_data, valid_data = data[:n], data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data inputs x and targets y
    data = train_data if split == 'train' else valid_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # starting index for each sequence
    x = torch.stack([data[i:i+block_size] for i in ix]) # input data
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # target data
    x, y = x.to(device), y.to(device) # move data to device
    return x, y

@torch.no_grad()
def estiamte_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# self attention head 
class Head(nn.Module):
    """ one head of self attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, H)
        q = self.query(x) # (B, T, H)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, H) @ (B, H, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # randomly prevent some tokens to communicate with each other
        # perform the weighted aggregation of values
        v = self.value(x) # (B, T, H)
        out = wei @ v # (B,T,T) @ (B, T, H) -> (B, T, H)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # num_heads * head_size = n_embd
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # projection back to original pathway (residual x+f(x)) + information mixing from all heads
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a nonlinearity """

    def __init__(self, n_embd):
        super().__init__()
        # position wise feed forward network
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # expand the dimensionality by a factor of 4 - expansion layer
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # this is projection layer back to original pathway - contraction layer
            nn.Dropout(dropout) # dropout layer - right before connection back to original pathway (residual x+f(x))
        )
    
    def forward(self, x):
        return self.net(x)
    

# Now we want blocks where tokens communicate with each other and then process themselves
class Block(nn.Module):
    """ Transformer block: communication followed by computation/processing """

    def __init__(self, n_embd, n_head):
        """Initializer for the Block class

        Args:
            n_embd: embedding dimension
            n_head: the number of heads we'd like 
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # by adding the input to the output of the self-attention and feed forward layers, 
        # we are allowing the model to learn residual connections like in ResNet
        x = x + self.sa(self.ln1(x)) # communication
        x = x + self.ffwd(self.ln2(x)) # computation/processing
        return x


#super simple Transformer model
class TransformerLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #self.sa_heads = MultiHeadAttention(num_heads=4, head_size=n_embd//4) # i.e. 4 heads of 8-dimensional self-attention (32/4)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd)
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C) - batch size, time steps, channels(n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        # x = self.sa_heads(x) # (B, T, C) apply one head of self-attention. # (so here tokens communicate with each other)!!
        # x = self.ffwd(x) # (B, T, C) apply feed forward layer # (and here each token is processing itself, what it learned from other tokens)!!!
        x = self.blocks(x) # (B, T, C) apply blocks of self-attention and feed forward, where tokens communicate with each other and then process themselves
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, V) - batch size, time steps, vocab size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B* T, C)
            targets = targets.view(-1) # flatten the targets B*T
            loss = F.cross_entropy(logits, targets) # cross entropy needs input as (B, C, T) and target as (B, T), where T can be many dim, but C needs to be 2nd dim

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) tensor of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            id_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled token to the running sequence
            idx = torch.cat([idx, id_next], dim=1) # (B, T+1)
        return idx

model = TransformerLanguageModel()
model = model.to(device)
print(f'Num Parameters: {sum(p.numel() for p in model.parameters())}')

 
# create a torch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
import time
start_time = time.time()

for iter in range(max_iters):

    #every once in a while, evaluate the loss on train and validation data
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estiamte_loss()
        print(f'iter {iter}, train loss: {losses["train"]:.4f}, valid loss: {losses["valid"]:.4f}')
        elapsed = time.time() - start_time
        print(f'{elapsed / (iter + 1):.4f} sec per iteration')

    # sample a batch of data
    xb, yb = get_batch('train')

     # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print('Training done!')
print(f'elapsed time: {time.time() - start_time:.2f} seconds')

# Save the model's state dict
torch.save(model.state_dict(), 'transformer_model.pt')
print("Model saved as 'transformer_model.pt'.")

# To load the model (for later use or inference), you can use:
model = TransformerLanguageModel()
model.load_state_dict(torch.load('transformer_model.pt', map_location=device))
model = model.to(device)
print("Model loaded from 'transformer_model.pt'.")

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, 500).squeeze().tolist()))