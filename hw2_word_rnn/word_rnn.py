import os
import re

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)

DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DIR, "saltan.txt"), encoding="utf-8") as file:
    text = file.read()

# разбиваем текст на слова вместо символов
words = re.findall("[а-яё]+", text.lower())
vocab = sorted(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
print(f"слов в тексте: {len(words)}, уникальных: {len(vocab)}")

SEQ_LEN = 6

# обучающие пары: SEQ_LEN слов -> следующее слово
encoded = torch.tensor([word2idx[w] for w in words])
X = torch.stack([encoded[i:i + SEQ_LEN] for i in range(len(encoded) - SEQ_LEN)])
Y = encoded[SEQ_LEN:]


class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        # каждое слово превращается в вектор, а не в one-hot на весь словарь
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.input_hidden = nn.Linear(embed_size + hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.hidden_output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) индексы слов
        emb = self.embedding(x)
        hidden = torch.zeros(x.shape[0], self.hidden_size)
        for t in range(x.shape[1]):
            hidden = self.activation(self.input_hidden(torch.cat((emb[:, t], hidden), dim=1)))
        return self.hidden_output(hidden)


def train(model, X, Y, epochs=40, batch_size=64, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses, accs = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X))
        epoch_loss, correct = 0.0, 0
        for start in range(0, len(X), batch_size):
            batch = perm[start:start + batch_size]
            logits = model(X[batch])
            loss = criterion(logits, Y[batch])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch)
            correct += (logits.argmax(dim=1) == Y[batch]).sum().item()

        losses.append(epoch_loss / len(X))
        accs.append(correct / len(X))
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | loss {losses[-1]:.4f} | accuracy {accs[-1]:.4f}")

    return losses, accs


def predict(model, phrase):
    model.eval()
    seq = [word2idx[w] for w in phrase.lower().split()][-SEQ_LEN:]
    with torch.no_grad():
        logits = model(torch.tensor([seq]))
    return idx2word[logits.argmax(dim=1).item()]


def generate(model, phrase, n_words=30, temperature=0.8):
    model.eval()
    result = phrase.lower().split()
    with torch.no_grad():
        for _ in range(n_words):
            seq = [word2idx[w] for w in result[-SEQ_LEN:]]
            logits = model(torch.tensor([seq]))[0] / temperature
            probs = torch.softmax(logits, dim=0)
            result.append(idx2word[torch.multinomial(probs, 1).item()])
    return " ".join(result)


def plot(losses, accs):
    os.makedirs(os.path.join(DIR, "images"), exist_ok=True)
    epochs = range(1, len(losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, losses)
    ax1.set_title("Loss (cross-entropy)")
    ax1.set_xlabel("epoch")
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, accs)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epoch")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, "images", "loss_accuracy.png"), dpi=120)
    plt.show()


model = WordRNN(len(vocab))
losses, accs = train(model, X, Y)

phrase = "три девицы под окном пряли поздно"
print(f"\n{phrase} -> {predict(model, phrase)}")
print("\nсгенерированный текст:")
print(generate(model, "ветер по морю гуляет и кораблик"))

plot(losses, accs)
