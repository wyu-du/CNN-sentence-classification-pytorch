from model import CNN
import utils

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy
import os


def train(data, params):
    if params["MODEL"] != "rand":
        # load word2vec
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

    model = CNN(**params).cuda(params["GPU"])

#    parameters = filter(lambda p: p.requires_grad, model.parameters())
#    optimizer = optim.Adadelta(model.parameters(), params["LEARNING_RATE"])
    optimizer = optim.Adam(model.parameters(), params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_acc = 0
    max_dev_acc = 0
    max_test_acc = 0
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]
            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]

            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=params["NORM_LIMIT"])
            optimizer.step()

        dev_acc = test(data, data, model, params, mode="dev")
        test_acc = test(data, data, model, params)
        print("epoch:", e + 1, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc)

        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_dev_acc = dev_acc

        if dev_acc > max_dev_acc:
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            best_model = copy.deepcopy(model)

    print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
    return best_model


def word_to_idx(ori_data, w):
    out_idx = 0
    if w in ori_data["word_to_idx"].keys():
        out_idx = ori_data["word_to_idx"][w]
    else:
        out_idx = ori_data["word_to_idx"]['[UNK]']
    return out_idx


def test(ori_data, cur_data, model, params, mode="test"):
    model.eval()

    if mode == "dev":
        x, y = ori_data["dev_x"], ori_data["dev_y"]
    elif mode == "test":
        x, y = cur_data["test_x"], cur_data["test_y"]
        
    correct = 0.
    counts = 0.
    for i in range(0, len(x), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(x) - i)

        batch_x = [[word_to_idx(ori_data, w) for w in sent] +
                   [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                   for sent in x[i:i + batch_range]]
        batch_y = [ori_data["classes"].index(c) for c in y[i:i + batch_range]]

        batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
        batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])

        pred = torch.argmax(model(batch_x), axis=1)
        acc = sum([1 if p == y else 0 for p, y in zip(pred, batch_y)])
        correct += acc
        counts += len(pred)

    return correct/counts


def predict(ori_data, cur_data, model, params, model_name="Seq2Seq"):
    model.eval()
    
    if model_name == 'test':
        x = cur_data["test_x"]
    else:
        x = cur_data[model_name]
    outs = []
    for i in range(0, len(x), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(x) - i)
        
        
        batch_x = [[word_to_idx(ori_data, w) for w in sent] +
                   [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                   for sent in x[i:i + batch_range]]
            

        batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
        pred = model(batch_x)
#        if model_name == 'test':
#            pred = F.softmax(pred, dim=1)
#            pred = pred.cpu().detach().numpy()
#            for b in range(len(pred)):
#                outs.append(pred[b])
        pred = torch.argmax(pred, axis=1)
        pred = pred.cpu().detach().numpy()
        for b in range(len(pred)):
            outs.append(ori_data["classes"][pred[b]])
    return outs


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="non-static", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--model_name", default="test", help="available models: Seq2Seq, HRED, VHRED, HRAN")
    parser.add_argument("--dataset", default="DA_Switchboard_sent", help="available datasets: MR, TREC, DailyDialog")
    parser.add_argument("--save_model", default=True, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=20, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument("--gpu", default=0, type=int, help="the number of gpu to be used")

    options = parser.parse_args()
    if options.mode == "model_pred":
        data = getattr(utils, f"read_DailyDialog_pred")()
    else:
        data = getattr(utils, f"read_{options.dataset}")()

    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"]+data["dev_y"]+data["test_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 64,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        if params["SAVE_MODEL"]:
            utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    elif options.mode == "test":
        model = utils.load_model(params).cuda(params["GPU"])
        test_acc = test(data, data, model, params)
        print("test acc (in domain):", test_acc)
        data_out = utils.read_DA_DialogBank_sent()
        test_acc = test(data, data_out, model, params)
        print("test acc (out of domain):", test_acc)
    elif options.mode == "model_pred":
        model = utils.load_model(params).cuda(params["GPU"])
        model_preds = predict(data, model, params, options.model_name)
        fpath = 'data/DA_ISO_sent/'+options.dataset+'_pred/'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        with open(fpath + options.model_name + '_label.txt', 'w') as f:
            for pred in model_preds:
                f.write(pred+'\n')
    else:
        model = utils.load_model(params).cuda(params["GPU"])
        model_preds = predict(data, data, model, params, options.model_name)
        fpath = 'data/DA_ISO_sent/'+options.dataset+'_pred/'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        with open(fpath + 'test_label.txt', 'w') as f:
            for pred in model_preds:
                f.write(pred+'\n')
        data_out = utils.read_DA_DialogBank_sent()
        model_preds_out = predict(data, data_out, model, params, options.model_name)
        with open(fpath + 'dialogbank_label.txt', 'w') as f:
            for pred in model_preds_out:
                f.write(pred+'\n')
    
if __name__ == "__main__":
    main()
