from sklearn.utils import shuffle

import pickle


def read_TREC():
    data = {}

    def read(mode):
        x, y = [], []

        with open("data/TREC/TREC_" + mode + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                y.append(line.split()[0].split(":")[0])
                x.append(line.split()[1:])

        x, y = shuffle(x, y)

        if mode == "train":
            dev_idx = len(x) // 10
            data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
            data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("test")

    return data


def read_DailyDialog():
    data = {}

    def read(mode):
        x, y = [], []

        with open("data/DailyDialog/intent_classification_" + mode + ".txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                if len(line) > 0:
                    y.append(line.split()[0])
                    x.append(line.split()[1:])

        x, y = shuffle(x, y)

        if mode == "train":
            data["train_x"], data["train_y"] = x, y
        elif mode == "dev":
            data["dev_x"], data["dev_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("dev")
    read("test")

    return data


def read_DailyDialog_sent():
    data = {}

    def read(mode):
        x, y = [], []

        with open("data/DailyDialog_sent/intent_sent_classification_" + mode + ".txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                if len(line) > 0:
                    y.append(line.split()[0])
                    x.append(line.split()[1:])

        x, y = shuffle(x, y)

        if mode == "train":
            data["train_x"], data["train_y"] = x, y
        elif mode == "dev":
            data["dev_x"], data["dev_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("dev")
    read("test")

    return data


def read_DA_DailyDialog_sent():
    data = {}

    def read(mode):
        x, y = [], []
        with open("data/DA_ISO_sent/" + mode + ".txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                if len(line) > 0:
                    y.append(line.split()[0])
                    x.append(line.split()[1:])
#        x, y = shuffle(x, y)
        if mode.split('_')[1] == "train":
            data["train_x"], data["train_y"] = x, y
        elif mode.split('_')[1] == "dev":
            data["dev_x"], data["dev_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("DailyDialog_train")
    read("DailyDialog_dev")
    read("DailyDialog_test")

    return data


def read_DA_AMI_sent():
    data = {}

    def read(mode):
        x, y = [], []
        with open("data/DA_ISO_sent/" + mode + ".txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                if len(line) > 0:
                    y.append(line.split()[0])
                    x.append(line.split()[1:])
#        x, y = shuffle(x, y)
        if mode.split('_')[1] == "train":
            data["train_x"], data["train_y"] = x, y
        elif mode.split('_')[1] == "dev":
            data["dev_x"], data["dev_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("AMI_train")
    read("AMI_dev")
    read("AMI_test")

    return data


def read_DA_MapTask_sent():
    data = {}

    def read(mode):
        x, y = [], []
        with open("data/DA_ISO_sent/" + mode + ".txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                if len(line) > 0:
                    y.append(line.split()[0])
                    x.append(line.split()[1:])
#        x, y = shuffle(x, y)
        if mode.split('_')[1] == "train":
            data["train_x"], data["train_y"] = x, y
        elif mode.split('_')[1] == "dev":
            data["dev_x"], data["dev_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("MapTask_train")
    read("MapTask_dev")
    read("MapTask_test")

    return data


def read_DA_Switchboard_sent():
    data = {}

    def read(mode):
        x, y = [], []
        with open("data/DA_ISO_sent/" + mode + ".txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                if len(line) > 0:
                    y.append(line.split()[0])
                    x.append(line.split()[1:])
#        x, y = shuffle(x, y)
        if mode.split('_')[1] == "train":
            data["train_x"], data["train_y"] = x, y
        elif mode.split('_')[1] == "dev":
            data["dev_x"], data["dev_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("Switchboard_train")
    read("Switchboard_dev")
    read("Switchboard_test")

    return data


def read_DA_All_sent():
    data = {}

    def read(mode):
        x, y = [], []
        with open("data/DA_ISO_sent/" + mode + ".txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                if len(line) > 0:
                    y.append(line.split()[0])
                    x.append(line.split()[1:])
#        x, y = shuffle(x, y)
        if mode.split('_')[1] == "train":
            data["train_x"], data["train_y"] = x, y
        elif mode.split('_')[1] == "dev":
            data["dev_x"], data["dev_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("All_train")
    read("All_dev")
    read("All_test")

    return data


def read_DA_DialogBank_sent():
    data = {}

    def read(mode):
        x, y = [], []
        with open("data/DA_ISO_sent/" + mode + ".txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                if len(line) > 0:
                    y.append(line.split()[0])
                    x.append(line.split()[1:])
#        x, y = shuffle(x, y)
        data["test_x"], data["test_y"] = x, y

    read("DialogBank_test")

    return data

def read_DailyDialog_pred():
    data = read_DailyDialog_sent()

    def read(model):
        x = []

        with open("data/DailyDialog_pred/" + model + ".txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                if len(line) > 0:
                    x.append(line.split())

        data[model] = x

    read("Seq2Seq")
    read("HRED")
    read("VHRED")
    read("HRAN")
    read("DSHRED_RA")
    read("test")

    return data


def read_MR():
    data = {}
    x, y = [], []

    with open("data/MR/rt-polarity.pos", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("data/MR/rt-polarity.neg", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

    return data


def save_model(model, params):
    path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    pickle.dump(model, open(path, "wb"))
    print(f"A model is saved successfully as {path}!")


def load_model(params):
    path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"

    try:
        model = pickle.load(open(path, "rb"))
        print(f"Model in {path} loaded successfully!")

        return model
    except:
        print(f"No available model such as {path}.")
        exit()
