import torch
import torch.nn as nn

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class BoWModel(nn.Module):
    def __init__(self, vocab_size, beo_size, mean, std):
        super(BoWModel, self).__init__()
        self.vocab_size = vocab_size

        self.bow_h = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.LeakyReLU(0.2),
        )

        self.bow_pred = nn.Sequential(
            nn.Linear(256, 5)
        )

        self.bow_embd = nn.Sequential(
            nn.Linear(256, 64)
        )

        self.beo_h = nn.Sequential(
            nn.Linear(beo_size, 256),
            nn.LeakyReLU(0.2),
        )

        self.beo_pred = nn.Sequential(
            nn.Linear(256, 5)
        )

        self.beo_embd = nn.Sequential(
            nn.Linear(256, 64)
        )

        self.mean = mean
        self.std = std

    def forward(self, s, x):
        # standardize
        x = (x - self.mean) / self.std

        # bow
        bow = Tensor(s.shape[0], self.vocab_size).zero_()
        bow.scatter_(1, s, 1)

        # hidden layer
        bow_h = self.bow_h(bow)
        beo_h = self.beo_h(x)

        # attribute prediction
        bow_p = self.bow_pred(bow_h)
        beo_p = self.beo_pred(beo_h)

        # embedding
        bow_e = self.bow_embd(bow_h)
        beo_e = self.beo_embd(beo_h)

        return (bow_e, beo_e, bow_p, beo_p)

class EmbedModel(nn.Module):
    def __init__(self, vocab_size, beo_size, embed_size, mean, std):
        super(EmbedModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.bow_h = nn.Sequential(
            nn.Linear(embed_size, 256),
            nn.LeakyReLU(0.2),
        )

        self.bow_pred = nn.Sequential(
            nn.Linear(256, 6)
        )

        self.bow_embd = nn.Sequential(
            nn.Linear(256, 64)
        )

        self.beo_h = nn.Sequential(
            nn.Linear(beo_size, 256),
            nn.LeakyReLU(0.2),
        )

        self.beo_pred = nn.Sequential(
            nn.Linear(256, 6)
        )

        self.beo_embd = nn.Sequential(
            nn.Linear(256, 64)
        )

        self.mean = mean
        self.std = std

    def forward(self, sentence, x):
        # standardize
        x = (x - self.mean) / self.std

        embedded = self.embed(sentence)
        avgEmbed = embedded.mean(dim=1)

        # hidden layer
        bow_h = self.bow_h(avgEmbed)
        beo_h = self.beo_h(x)

        # attribute prediction
        bow_p = self.bow_pred(bow_h)
        beo_p = self.beo_pred(beo_h)

        # embedding
        bow_e = self.bow_embd(bow_h)
        beo_e = self.beo_embd(beo_h)

        return (bow_e, beo_e, bow_p, beo_p)
