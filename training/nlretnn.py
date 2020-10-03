import csv
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import sys
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.word_to_vector import GloVe
import string
import itertools
from nlmodel import BoWModel, EmbedModel
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--beo_size', type=int, default=300, help='size of each image dimension')

parser.add_argument('--traindata', type=str, default='data/couch_train.csv', help='data file')
parser.add_argument('--testdata', type=str, default='data/couch_test.csv', help='data file')
parser.add_argument('--devdata', type=str, default='data/couch_dev.csv', help='data file')
parser.add_argument('--objvectors', type=str, default='data/couch_full_obv_vecs_300.csv', help='beo vectors')
parser.add_argument('--model_output', type=str, help='model output name')

opt = parser.parse_args()

os.makedirs('models', exist_ok=True)

SZ = opt.beo_size
RET_SIZE = 3
RET_MULT = 1
EMBED_SZ = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def train(model, device, train_data, optimizer, epoch, writer):
    model.train()
    sumloss = 0.0
    for (id, bow, vec, val, bow_adjs, beo_adjs) in train_data:
        vec = torch.from_numpy(vec).to(device).float().unsqueeze(0)
        optimizer.zero_grad()

        bow_e, beo_e, bow_p, beo_p = model(bow, vec)
        bow_target = Tensor(bow_adjs).unsqueeze(0)
        beo_target = Tensor(beo_adjs).unsqueeze(0)
        loss = (F.binary_cross_entropy_with_logits(bow_p, bow_target) \
             + F.binary_cross_entropy_with_logits(beo_p, beo_target) \
             + F.cosine_embedding_loss(bow_e, beo_e, Tensor([val]))) / 3.0
        loss.backward()
        optimizer.step()
        sumloss += loss.item()
    print()
    print(('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, sumloss / len(train_data))))
    writer.add_scalar('runs/train_loss', sumloss / len(train_data), epoch)

def eval(model, device, test_data, writer, epoch, prefix):
    model.eval()
    test_loss = 0.0
    sum_sim = 0.0
    correct = 0.0
    bow_truth = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    beo_truth = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    bow_pred = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    beo_pred = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    with torch.no_grad():
        for (id, bow, vec, val, bow_adjs, beo_adjs) in test_data:
            vec = torch.from_numpy(vec).to(device).float().unsqueeze(0)
            bow_e, beo_e, bow_p, beo_p = model(bow, vec)

            test_loss += F.cosine_embedding_loss(bow_e, beo_e, Tensor([val])).item() # sum up batch loss
            sim = F.cosine_similarity(bow_e, beo_e)
            sum_sim += sim.item()
            if (sim >= 0 and val >= 0) or (sim < 0 and val < 0):
                correct += 1.0

            # accuracy
            #print(beo_p)
            bow_p = binarize_prediction(bow_p[0])
            beo_p = binarize_prediction(beo_p[0])
            #print(beo_p)

            # store the true values and the predictions
            for i, v in enumerate(bow_adjs):
                bow_truth[i].append(v)
            for i, v in enumerate(beo_adjs):
                beo_truth[i].append(v)
            for i, v in enumerate(bow_p):
                bow_pred[i].append(v)
            for i, v in enumerate(beo_p):
                beo_pred[i].append(v)

    print('{}: AvgLoss: {} SumSim: {} Acc: {}'.format(prefix, test_loss / len(test_data), sum_sim / len(test_data), correct / len(test_data)))
    for i in range(6):
        print("ADJ ", i + 1)
        print("BOW preds")
        print(classification_report(bow_truth[i], bow_pred[i], target_names=["absent", "present"]))
        print("BEO preds")
        print(classification_report(beo_truth[i], beo_pred[i], target_names=["absent", "present"]))

    writer.add_scalar('runs/{}_loss'.format(prefix), test_loss / len(test_data), epoch)
    writer.add_scalar('runs/{}_accuracy'.format(prefix), correct / (len(test_data)), epoch)

def binarize_prediction(tensor):
    l = []
    for p in tensor:
        label = -1
        if p > 0.5:
            label = 1
        elif p < 0.5:
            label = 0
        else: #neutral average, randomize label
            label = random.choice([0,1])
        l.append(label)
    return l

#check if 2 lists of the same length are equal to each other
def equal(prediction, target):
    if (len(prediction) == 0 and len(target) == 0):
        return True
    elif prediction[0] != target[0]:
        return False
    else:
        return equal(prediction[1:], target[1:])

def retrieval(model, device, ret_data, writer, epoch):
    model.eval()
    correct = 0.0
    correct2 = 0.0
    with torch.no_grad():
        for (label, bow, vecs) in ret_data:
            vec = torch.from_numpy(vecs).to(device).float()
            bows = bow.repeat(len(vecs), 1)
            bow_e, beo_e, bow_p, beo_p = model(bows, vec)
            sim = F.cosine_similarity(bow_e, beo_e)
            sorted, indexes = sim.sort(descending=True)
            if 0 == indexes[0]:
                correct += 1
                correct2 += 1
            elif 0 == indexes[1]:
                correct2 += 1
    print('RET Top1: {} Top2: {}'.format(correct / len(ret_data), correct2 / len(ret_data)))
    writer.add_scalar('runs/top1_ret_accuracy', correct / (len(ret_data)), epoch)
    writer.add_scalar('runs/top2_ret_accuracy', correct2 / (len(ret_data)), epoch)

def makeRet(test_pos):
    test_ret = []
    for j in range(RET_MULT):
        for i in range(len(test_pos)):
            row = test_pos[i]
            vecs = [row[2]]
            while True:
                neg_sample = random.choice(test_pos)
                if not neg_sample[0] == row[0]:
                    vecs.append(neg_sample[2])
                if len(vecs) >= RET_SIZE:
                    break
            test_ret.append((row[0], row[1], np.array(vecs)))
    return test_ret

def SentenceToTensor(s, encoder):
    t = encoder.encode(s)
    inp = t % encoder.vocab_size
    inp = inp.unsqueeze(0).to(device)

    return inp

def csvRowsToDataset(datarows, id2vec):
    avgLabel = {}
    for row in datarows:
        for i in range(6):
            if not row[0] in avgLabel:
                avgLabel[row[0]] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            avgLabel[row[0]][i] += int(row[i+1]) / 10.0

    dataset = []
    for row in datarows:
        dataset.append((row[0],
                        row[7].translate(str.maketrans('', '', string.punctuation)).lower(),
                        id2vec[row[0]],
                        list(map(binarize, avgLabel[row[0]]))))
    return dataset

def binarize(x):
    label = -1
    if x > 3:
        label = 1
    elif x < 3:
        label = 0
    else: #neutral average, randomize label
        label = random.choice([0,1])
    return label

# data is (id, sentence, beo_vec)
def main():
    # set the random seeds
    random.seed(0)
    torch.manual_seed(0)

    with open(opt.traindata, 'r') as traindatafile:
        with open(opt.devdata, 'r') as devdatafile:
            with open(opt.testdata, 'r') as testdatafile:
                with open(opt.objvectors, 'r') as vectorfile:
                    traindatarows = list(csv.reader(traindatafile))
                    devdatarows = list(csv.reader(devdatafile))
                    testdatarows = list(csv.reader(testdatafile))
                    vectorrows = csv.reader(vectorfile)

                    # make a dict of ids to numpy beo vecs
                    make_vector = lambda s: np.fromstring(s.replace('\n', '')[1:-1], dtype=np.float, sep=' ')
                    id2vec = {row[0] : make_vector(row[1]) for row in vectorrows}

                    # find the avgs of the labelings
                    train_data = csvRowsToDataset(traindatarows, id2vec)
                    dev_data = csvRowsToDataset(devdatarows, id2vec)
                    test_data = csvRowsToDataset(testdatarows, id2vec)
                    encoder = WhitespaceEncoder([row[1] for row in train_data], min_occurrences=1)

                    print("Vocab Size", encoder.vocab_size)

                    # train data
                    train_data = [(row[0], SentenceToTensor(row[1], encoder), row[2], row[3]) for row in train_data]

                    # find mean and standard deviation
                    meanVec = Tensor(np.mean(np.array(list(map(lambda row: row[2], train_data))), axis=0))
                    stdVec = Tensor(np.std(list(map(lambda row: row[2], train_data)), axis=0))

                    train_pos = [(row[0], row[1], row[2], 1.0, row[3], row[3]) for row in train_data]

                    train_neg = []
                    for row in train_pos:
                        while True:
                            neg_sample = random.choice(train_pos)
                            if not neg_sample[0] == row[0]:
                                train_neg += [(row[0], neg_sample[1], row[2], -1.0, neg_sample[4], row[4])]
                                break
                    train_data = train_pos + train_neg

                    # dev data
                    dev_data = [(row[0], SentenceToTensor(row[1], encoder), row[2], row[3]) for row in dev_data]
                    dev_pos = [(row[0], row[1], row[2], 1.0, row[3], row[3]) for row in dev_data]

                    dev_neg = []
                    for i in range(len(dev_pos)):
                      row = dev_pos[i]
                      while True:
                          neg_sample = random.choice(dev_pos)
                          if not neg_sample[0] == row[0]:
                              dev_neg += [(row[0], neg_sample[1], row[2], -1.0, neg_sample[4], row[4])]
                              break
                    dev_data = dev_pos + dev_neg

                    # test data
                    test_data = [(row[0], SentenceToTensor(row[1], encoder), row[2], row[3], row[3]) for row in test_data]
                    test_pos = [(row[0], row[1], row[2], 1.0, row[3], row[3]) for row in test_data]

                    test_neg = []
                    for i in range(len(test_pos)):
                        row = test_pos[i]
                        while True:
                            neg_sample = random.choice(test_pos)
                            if not neg_sample[0] == row[0]:
                                test_neg += [(row[0], neg_sample[1], row[2], -1.0, neg_sample[4], row[4])]
                                break
                    test_data = test_pos + test_neg
                    ret_data = makeRet(test_pos)

                    print(len(train_data), len(dev_data), len(test_data), len(ret_data))
                    model = EmbedModel(encoder.vocab_size, opt.beo_size, EMBED_SZ, meanVec, stdVec).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=0.0001)

                    # load word embeddings
                    word_vectors = GloVe(name="6B", dim=str(EMBED_SZ))
                    for i, token in enumerate(encoder.vocab):
                        model.embed.weight.data[i] = word_vectors[token]

                    writer = SummaryWriter()
                    eval(model, device, dev_data, writer, 0, "Dev")
                    eval(model, device, test_data, writer, 0, "Test")
                    retrieval(model, device, ret_data, writer, 0)
                    for epoch in range(0, 100):
                        print("Epoch ", epoch + 1)
                        random.shuffle(train_data)
                        train(model, device, train_data, optimizer, epoch, writer)
                        eval(model, device, dev_data, writer, epoch, "Dev")
                        eval(model, device, test_data, writer, epoch, "Test")
                        retrieval(model, device, ret_data, writer, epoch)
                    torch.save(model.state_dict(), os.path.join('.', 'models', opt.model_output, 'nlmodel.pt'))

if __name__ == "__main__":
    main()
