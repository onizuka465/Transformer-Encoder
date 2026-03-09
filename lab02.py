import pandas as pd
import numpy as np 

vocab = {"A":0, "Banana":1, "é":2, "boa":3,"e":4, "barata":5}
df_vocab = pd.DataFrame (list(vocab.items()), columns= ["palavra","id"] )

frase = ["A","Banana", "é", "boa", "e", "barata"]
ids = [vocab[palavra] for palavra in frase]

d_model=64
vocab_size= len(vocab)
emmbeding_table = np.random.randn(vocab_size, d_model)

X = emmbeding_table[ids]
X = X[np.newaxis, :, :]

WQ = np.random.randn(d_model, d_model)
WK = np.random.randn(d_model, d_model)
WV = np.random.randn(d_model, d_model)

d_ff = 256

W1 = np.random.randn(d_model,d_ff)
W2 = np.random.randn (d_ff, d_model)
b1 = np.zeros(d_ff)
b2 = np.zeros(d_model)

#funções


def softmax(x):
    e_x = np.exp(x)     
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean)/ np.sqrt(var + epsilon)

def self_attention(x):
    Q = x @ WQ
    K = x @ WK
    V = x @ WV
    K_T = K.transpose (0, 2, 1)
    scores = Q @ K_T
    scores_scaled = scores / np.sqrt(d_model)
    weigths = softmax(scores_scaled)
    return weigths @ V

def ffn(x):
    camada1 = np.maximum(0, x @ W1 + b1)
    camada2 = camada1 @ W2 + b2
    return camada2

for i in range (6):
    X_att = self_attention(X)
    X_norm1 = layer_norm(X + X_att)
    X_ffn = ffn(X_norm1)
    X_out = layer_norm(X_norm1 + X_ffn)
    X = X_out 

print("Shape final de Z:", X.shape)