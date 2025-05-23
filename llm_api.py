import os
import argparse
import logging
import datetime
import time
from accelerate import Accelerator
from LoRA_EA import GLM_Lora, Qwen_Lora, Qwen2vl_Lora, llama2_Lora, llama3_Lora

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

unalign = '999999'

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            datetime.timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


# Configure logging
class Logger:
    def __init__(self, log_dir, log_name=None):
        """
        Initialize logger object
        :param log_dir: Directory to save log files
        :param log_name: Log file name (optional, defaults to current time)
        """
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # create log formatter
        log_formatter = LogFormatter()

        # Set log file name
        if log_name is None:
            log_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        else:
            log_name += datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        self.log_filepath = os.path.join(log_dir, log_name)

        # create console handler and set level to info
        filepath, file_handler = None, None
        if self.log_filepath is not None:
            filepath = '%s' % (self.log_filepath)
            file_handler = logging.FileHandler(self.log_filepath, "a", encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(log_formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)

        self.logger = logging.getLogger()
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if filepath is not None:
            self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        """Log information message"""
        self.logger.info(message)

    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message):
        """Log error message"""
        self.logger.error(message)


class cfg():
    def __init__(self):
        self.args = None

    def get_args(self):
        parser = argparse.ArgumentParser()
        # basewith_tf
        parser.add_argument('--lora', action="store_true", default=False, help="LoRA LLM")
        parser.add_argument('--test', action="store_true", default=False, help="test model")
        parser.add_argument('--neg_sample', default=1, type=int, choices=[1, 2, 5], help="Negative sampling ratio, 1- Hard negative sampling only")
        parser.add_argument('--dropout', default=0.2, type=float, help="Suggested value: 0.2-0.3")
        parser.add_argument('--epoch', default=1, type=int)
        parser.add_argument('--dataset', default='FBYG', type=str,
                            choices=["FBYG", "FBDB", "EN_FR_15K_V2", "EN_DE_15K_V2", "ja_en", "zh_en", "de_en"],
                            help="Strategy of information Filter")
        parser.add_argument('--strage', default='m_l', type=str, choices=["l_np", "m_l", "m_l_n", "m_l_np", "m_np", "ma_l_np", "matf", "matf_l_np", "matf_l", "mr_l_np", "MM_matf", "MM_matf_l_np"], help="Strategy of information Filter")
        self.args = parser.parse_args()


def get_topk(result, s2t, logger):
    import numpy as np
    import torch
    top_k = [1, 3, 5, 10, 50]
    nofind = 0
    acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
    test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
    for tar, res in result.items():
        ca = torch.tensor(res)
        try:
            rank = (ca == s2t[int(tar)]).nonzero(as_tuple=False).squeeze().item()
        except:
            nofind += 1
            continue
        mean_l2r += (rank + 1)
        mrr_l2r += 1.0 / (rank + 1)
        for i in range(len(top_k)):
            if rank < top_k[i]:
                acc_l2r[i] += 1
    mean_l2r /= len(result)
    mean_r2l /= len(result)
    mrr_l2r /= len(result)
    mrr_r2l /= len(result)
    for i in range(len(top_k)):
        acc_l2r[i] = round(acc_l2r[i] / len(result), 4)
    logger.info(f"rank result:His@k[1, 3, 5, 10]:{acc_l2r} mrr: {round(mrr_l2r, 4)}")
    logger.info(f"no alignment:{nofind}")


def rerank(ill_path, pre_path, result_path, logger):
    """
    target - align
    :param predict: [{"target": target, "align": output, "predict": predict, "candidates": candis}]
    :param result_path: path, result: {str(id): [candidate_entities]}
    :return:
    """
    import json
    with open(pre_path, 'r') as f:
        predicts = json.load(f)
    with open(result_path, 'r') as f:
        result = json.load(f)

    def read_file(file_paths):
        tups = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    tups.append(tuple([int(x) for x in params]))
        return tups

    ills = read_file([ill_path + "/ill_ent_ids"])
    s2t = {}
    t2s = {}
    for i, j in ills:
        s2t[i] = j
        t2s[j] = i

    # In FB2DB15K, there are 2 entities in DB that align with 2 entities in FB respectively
    # Maintain original order while removing duplicate elements - reason: duplicate elements are adjacent in sequence, removing them doesn't affect actual results

    if len(list(result.values())[0]) != len(set(list(result.values())[0])):
        for tar, res in result.items():
            seen = {}  # Used to record whether elements have appeared
            new_res = []  # Used to store deduplicated results
            for r in res:
                if r not in seen:
                    seen[r] = 1  # First appearance
                    new_res.append(r)
                else:
                    seen[r] += 1  # Second appearance, skip
            result[tar] = new_res

    logger.info("test result")
    get_topk(result, s2t, logger)
    mid = 0
    found = 0
    noin = 0
    npre = 0
    acc = 0
    rerank_res = {}
    for predict in predicts:
        tar, lab, pre, ca, _ = predict.values()
        if pre == lab and lab != unalign:
            acc += 1
        try:
            res = result[str(tar)]
        except:
            noin += 1
            continue
        if pre != unalign and pre in ca:
            try:
                index = res.index(int(pre))
            except:
                npre += 1
                continue
            if index != 0:
                # Maintain original order, insert aligned entity at first position
                res.pop(index)
                res.insert(0, int(pre))
            mid += index + 1
            found += 1
            rerank_res[str(tar)] = res
        else:
            rerank_res[str(tar)] = res
    logger.info('Number of targets not in source entities: {0}'.format(noin))
    logger.info('Number of targets not in candidate entities: {0}'.format(npre))
    get_topk(rerank_res, s2t, logger)
    logger.info("Accuracy: {0}".format(round(acc / len(predicts), 4)))
    logger.info("Total test data count: {0}".format(len(predicts)))
    try:
        logger.info("rerank mean of index: {0}".format(round(mid / found, 4)))
    except:
        print("rerank sum of index:", mid, found)
        logger.info("no find")


if __name__ == '__main__':
    cfg = cfg()
    cfg.get_args()
    # Initialize Accelerator
    accelerator = Accelerator()

    basemodel = "PMF"
    il = "nil"
    num = 10
    model = 'GLM4'
    root_path = "/model_path/"

    if model == 'GLM4':
        model_path = root_path + "ZhipuAI/glm-4-9b-chat/"
        lora_path = 'GLM4_'
        llm = GLM_Lora(model_path)

    elif model == 'Qwen2':
        model_path = root_path + "Qwen/Qwen2-7B/"
        lora_path = 'Qwen2_'
        llm = Qwen_Lora(model_path)

    elif model == 'Qwen2vl':
        model_path = root_path + "Qwen/Qwen2-VL-7B-Instruct/"
        lora_path = 'Qwen2vl_'
        llm = Qwen2vl_Lora(model_path)

    else:
        print('No base model selected')
        exit()

    for rate in ["0.2", "0.5", "0.8"]:
        logger = Logger('./logs', basemodel + '_' + '0.2' + '_' + il + "_p1")
        start = 0
        end = 10
        data_paths = []
        if basemodel != 'MEAformer':
            dataset1 = basemodel + '_' + cfg.args.dataset
        else:
            dataset1 = cfg.args.dataset
        if 'V2' in cfg.args.dataset:
            dataset1 = dataset1 + '_norm'
        elif 'en' in cfg.args.dataset:
            dataset1 = basemodel + '_DBP15K_' + cfg.args.dataset
        else:
            dataset1 = dataset1 + '15K'
        for i in range(5):
            data_paths.append(
                "result/{0}/{1}/{2}_{3}_{4}/instruct_zero_{5}-{6}".format(basemodel, il, cfg.args.dataset, rate,
                                                                          cfg.args.strage, start, end))
            start = end
            end += 10
        print(data_paths)
        lora_path = lora_path + basemodel + '_' + cfg.args.dataset + '_' + il + "_p1_" + cfg.args.strage

        if cfg.args.lora:
            logger.info('train')
            logger.info('lora_dropout: {0}, num_train_epochs: {1}, strategy: {2}'.format(cfg.args.dropout, cfg.args.epoch,
                                                                                       cfg.args.strage))
            print([data_path + "_train.json" for data_path in data_paths[:cfg.args.neg_sample]])
            llm.model_lora([data_path + "_train.json" for data_path in data_paths[:cfg.args.neg_sample]], lora_path,
                           cfg.args, accelerator)

        if cfg.args.test:
            logger.info('eval')
            logger.info('strategy: {0}'.format(cfg.args.strage))
            pre_path = "./result/{0}/{1}/{2}_{3}_{4}/eval_{5}_result.json".format(basemodel, il, cfg.args.dataset, rate,
                                                                                  cfg.args.strage, num)
            if 'l' in cfg.args.strage and 'm' in cfg.args.strage:
                test = cfg.args.strage.split('_l')[0]
            elif 'l' not in cfg.args.strage and 'm' in cfg.args.strage:
                test = 'matf'
            elif 'l' in cfg.args.strage and 'm' not in cfg.args.strage:
                test = 'none'
            else:
                test = 'none'
            llm.model_test([data_path.replace(cfg.args.strage, test) + "_eval.json" for data_path in data_paths],
                           lora_path + '_' + str(cfg.args.epoch), pre_path)

            result_path = "./result/{0}/{1}/{2}_{3}_{1}_left_{4}.json".format(basemodel, il, dataset1, rate,
                                                                              'eval')
            file_dir = 'data/mmkb/{0}'.format(cfg.args.dataset + '15K')
            rerank(file_dir, pre_path, result_path, logger)
