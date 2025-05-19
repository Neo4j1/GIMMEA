import json
import random
import re

import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from PIL import Image
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from qwen_vl_utils import process_vision_info
from torch.nn import functional as F  # 关键导入
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, \
    TrainerCallback, Qwen2VLForConditionalGeneration, AutoConfig
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, \
    AutoProcessor

# import numpy as np

# 获取可用的 GPU 数量
gpu_count = torch.cuda.device_count()
print(f"当前可用的 GPU 数量: {gpu_count}")

prompt = "You're an expert in knowledge graph alignment.The alignment of the target entity can be analyzed based on the degree of similarity of information between the candidate entity and the target entity."
prompt1 = "You're an expert in knowledge graph alignment."
prompt2 = "Suppose you are an expert in knowledge graph alignment."


def save_peft_model(trainer, tokenizer, lora_path):
    peft_model_id = lora_path
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    print("save model finished!")


def get_acc(predicts):
    acc = 0
    all = 0
    noin = 0
    rs = 0
    for predict in predicts:
        tar, align, pre, ca, r = predict.values()
        rs += r
        all += 1
        if str(align) in ca:
            if pre == str(align) and str(align) != "999999":
                acc += 1
            if str(align) == "999999":
                noin += 1
    print('小模型准确率：', 1 - noin / all)
    # print('微调模型准确率：', acc / all)
    print("accuary of align:", acc / all)
    print("Average round:", rs / all)
    print("num of align entity:", acc)
    # print('平均推理次数：', rs / all)


class SavePeftModelCallback(TrainerCallback):
    def __init__(self, trainer, tokenizer, base_save_path):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.base_save_path = base_save_path  # 基础保存路径

    def on_epoch_end(self, args, state, control, **kwargs):
        # 在每个 epoch 结束时动态生成保存路径，例如：./save_model_epoch_1, ./save_model_epoch_2
        save_path = f"{self.base_save_path}_{round(state.epoch)}"
        print('save model path:', save_path)
        save_peft_model(self.trainer, self.tokenizer, save_path)
        return control


def vectorized_find_sequence(b_tensor, a_list, eqal=True):
    if isinstance(a_list, torch.Tensor):
        # 方案1: 已有Tensor的复制
        a_tensor = a_list.clone().detach().to(b_tensor.device)
    else:
        # 方案2: 从Python数据创建
        a_tensor = torch.tensor(a_list, device=b_tensor.device)
    a_len = len(a_list)

    # 生成所有滑动窗口 [b_len - a_len + 1, a_len]
    windows = b_tensor.unfold(-1, a_len, 1)

    # 比较所有窗口 [..., num_windows]
    match_mask = (windows == a_tensor).all(-1)

    # 生成最终掩码
    result = torch.zeros_like(b_tensor, dtype=torch.bool)
    for i in range(match_mask.shape[-1]):
        if match_mask[..., i]:
            result[..., i:i + a_len] = True
            break
    if not eqal:
        if result.any():
            result = torch.zeros_like(b_tensor, dtype=torch.bool)
        else:
            result = torch.ones_like(b_tensor, dtype=torch.bool)
    return result


class AlignmentAwareLossTrainer(Trainer):
    def __init__(self, *args, unaligned_token_id, **kwargs):
        super().__init__(*args, **kwargs)
        self.unaligned_token_id = unaligned_token_id  # 未对齐标识的token ID
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.current_epoch = 0  # 新增：跟踪当前epoch

    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. 前向计算（禁用缓存以节省显存）
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False  # 必须关闭！
        )
        logits = outputs.logits
        # print('outputs', len(logits[0]), '\nlabels', len(inputs["labels"][0]))
        # valid_mask = inputs["labels"] != -100
        # valid_output = output[valid_mask.unsqueeze(-1).expand_as(output)].view(-1, output.size(-1))
        # valid_labels = inputs["labels"][valid_mask]  # [num_valid]
        # print(valid_labels, valid_mask)

        # for pos in range(inputs["input_ids"].shape[1]):
        #     if valid_mask[0][pos]:
        #         predicted_id = torch.argmax(outputs.logits[0, pos]).item()
        #         print(f"输入[{pos}]: {inputs['labels'][0][pos]} → 预测: {predicted_id}")
        if hasattr(self, 'state') and self.state is not None:
            self.current_epoch = self.state.epoch
        # 2. 分批计算损失（关键改进）
        # 修改：根据epoch选择损失计算方式
        if self.current_epoch <= 2:  # 前2个epoch只用CE
            loss = self.ce_loss(logits, inputs["labels"])
        else:  # 第3个epoch开始加入自定义损失
            loss = self._compute_memory_efficient_loss(
                logits=logits,
                labels=inputs["labels"],
                unaligned_token_id=self.unaligned_token_id
            )

        return (loss, outputs) if return_outputs else loss

    def _compute_memory_efficient_loss(self, logits, labels, unaligned_token_id):
        # 初始化总损失
        total_loss = torch.tensor(0.0, device=logits.device)

        # 3. 逐样本计算（避免矩阵操作显存爆炸）
        for i in range(logits.shape[0]):
            # 提取当前样本的logits和labels
            sample_logits = logits[i, :, :]  # [seq_len, vocab_size]
            sample_labels = labels[i, :]  # [seq_len]

            # 计算标准交叉熵（仅计算非padding位置）
            ce_loss = F.cross_entropy(
                sample_logits.view(-1, sample_logits.size(-1)),
                sample_labels.view(-1),
                ignore_index=-100
            )

            # 计算对齐惩罚项（仅关键位置）
            # align_mask = (sample_labels.unsqueeze(-1) != unaligned_token_id) & (sample_labels != -100)
            # unalign_mask = (sample_labels.unsqueeze(-1) == unaligned_token_id) & (sample_labels != -100)

            # 修改：第3个epoch后逐步引入自定义损失
            if self.current_epoch > 2:
                align_mask = vectorized_find_sequence(sample_labels, unaligned_token_id, eqal=False) & (
                        sample_labels != -100)
                unalign_mask = vectorized_find_sequence(sample_labels, unaligned_token_id) & (sample_labels != -100)

                align_penalty = torch.tensor(0.0, device=sample_logits.device)
                unalign_penalty = torch.tensor(0.0, device=sample_logits.device)
                reward = torch.tensor(0.0, device=sample_logits.device)

                if align_mask.any():
                    probs = F.softmax(sample_logits[align_mask], dim=-1)
                    align_penalty = probs[:, unaligned_token_id].mean()
                    correct_preds = vectorized_find_sequence(
                        torch.argmax(sample_logits[align_mask], dim=-1),
                        sample_labels[align_mask]
                    )
                    reward += correct_preds.float().mean()

                if unalign_mask.any():
                    probs = F.softmax(sample_logits[unalign_mask], dim=-1)
                    unalign_penalty = 1 - probs[:, unaligned_token_id].mean()
                    correct_preds = vectorized_find_sequence(
                        torch.argmax(sample_logits[unalign_mask], dim=-1),
                        unaligned_token_id
                    )
                    reward += correct_preds.float().mean()

                # 修改：渐进式加权（随epoch增加权重）
                custom_loss_weight = min(1.0, (self.current_epoch - 2) / 3.0)  # 第3-5epoch从0.33→1.0
                total_loss += ce_loss + custom_loss_weight * (
                        0.5 * (align_penalty + unalign_penalty) - 1.0 * reward
                )
                print('CE loss:', ce_loss.item(), 'align_penalty', align_penalty.item(), 'unalign_penalty',
                      unalign_penalty.item(), 'reward', reward.item())
            else:
                total_loss += ce_loss

        print('average loss:', total_loss / logits.shape[0])

        return total_loss / logits.shape[0]  # 返回批次平均损失


def model_param(model):
    names = []
    for name, param in model.named_parameters():
        if name not in names:
            names.append(name)
        else:
            print(name)
        # if param.requires_grad:
        #     print(f"{name}: requires_grad={param.requires_grad}")


def load_and_merge_json_files(file_paths):
    """读取多个JSON文件并按target字段合并分组"""
    all_data = []

    # 1. 读取所有文件数据
    for path in file_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            all_data.extend(data)  # 合并数据

    # 2. 按target分组
    target_groups = defaultdict(list)
    for item in all_data:
        target_groups[item["target"]].append(item)

    # 3. 按target排序后返回
    redata = [group for target, group in sorted(target_groups.items())]
    print(redata[0][0]["target"] == redata[0][1]["target"] == redata[0][2]["target"] == redata[0][3]["target"] == redata[0][4]["target"])
    return [group for target, group in sorted(target_groups.items())]


class GLM_Lora:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
        # self.tokenizer.add_special_tokens({"additional_special_tokens": ["[UNALIGNED]"]})  # 关键步骤！
        # print("[UNALIGNED]的ID:", self.tokenizer.convert_tokens_to_ids("[UNALIGNED]"))
        self.unaligned = '999999'
        self.train = True
        self.device = "cuda:1"
        self.prompt = prompt1
        self.output = "Output:"
        # self.output = "Align entity:"

    def process_func(self, example):
        MAX_LENGTH = 2048
        input_ids, attention_mask, labels = [], [], []
        instruction = self.tokenizer((f"[gMASK]<sop><|system|>\n{self.prompt}<|user|>\n"
                                      f"{example['instruction'] + example['input']}<|assistant|>\n"
                                      ).strip(),
                                     add_special_tokens=False)

        response = self.tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def model_lora(self, data_paths, lora_path, arg, accelerator=None):
        # 将JSON文件转换为CSV文件
        dfs = []
        for path in data_paths:
            df = pd.read_json(path)
            dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 使用 sample 打乱数据
        # print('数据总量', len(df))
        ds = Dataset.from_pandas(df)
        torch.cuda.empty_cache()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_id = ds.map(self.process_func, remove_columns=ds.column_names)

        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
        model_param(model)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            # task_type=TaskType.QUESTION_ANS,
            target_modules=["query_key_value"],
            inference_mode=False,  # 训练模式
            r=8,  # Lora 秩
            lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=arg.dropout  # Dropout 比例
        )
        args = TrainingArguments(
            output_dir="./output/GLM4",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            logging_steps=100,
            num_train_epochs=arg.epoch,
            save_strategy="epoch",
            learning_rate=1e-5,
            save_on_each_node=True,
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False  # 加速训练
        )

        model = get_peft_model(model, config)

        model.enable_input_require_grads()
        # model.gradient_checkpointing_enable()
        model.config.use_cache = False

        model.print_trainable_parameters()

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_id,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
        )

        if accelerator:
            # 使用 Accelerate 准备模型和数据
            model, trainer = accelerator.prepare(model, trainer)
        # 添加自定义回调函数
        # 每个批次的结果都保存
        # trainer.add_callback(SavePeftModelCallback(trainer, self.tokenizer, lora_path))

        trainer.train()
        # 只保存最后一个批次
        save_peft_model(trainer, self.tokenizer, lora_path + '_' + str(arg.epoch))

    # def model_test(self, datapaths, lora_path, dataset, basemodel, num, il):
    def model_test(self, datapaths, lora_path, save_path, accelerator=None):
        self.train = False
        torch.cuda.set_device(self.device)

        print("lora path:{0}\ndata path：{1}".format(lora_path, datapaths))
        # candi_ids_part = r'(?<=\{)\d+(?=\:)'
        candi_ids_part = r'\d+(?=\:)'

        # df = pd.read_json(datapaths)
        # dfs = []
        # for path in datapaths:
        #     df = pd.read_json(path)[:100]
        #     ds = Dataset.from_pandas(df)
        #     dfs.append(ds)
        dfs = load_and_merge_json_files(datapaths)
        print(len(dfs[0]), dfs[0][0]['target'] == dfs[0][1]['target'] == dfs[0][2]['target'] == dfs[0][3]['target'] == dfs[0][4]['target'])
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)

        # 加载lora权重
        model = PeftModel.from_pretrained(model, model_id=lora_path).to(self.device).eval()
        # model = model.to(self.device).eval()
        # self.tokenizer = AutoTokenizer.from_pretrained(lora_path, use_fast=True, trust_remote_code=True)

        # 将模型和数据分配到设备上
        # model, inputs = accelerator.prepare(model, dfs)

        acc = 0
        MAX_LENGTH = 4000
        tooLong = 0
        isNone = 0
        no_in_candi = 0
        acc_emb = 0
        savedata = []
        predicts = []
        # print(dfs[0].values[0])
        # for i in tqdm(range(10)):
        for i in tqdm(range(len(dfs))):
            rounds = 0
            for j in range(len(dfs[0])):
                ins, _, label, target = list(dfs[i][j].values())
                label = str(label)
                acc_emb += label != self.unaligned

                inputs = self.tokenizer.apply_chat_template(
                    [
                        {"role": "assistant", "content": self.prompt},
                        {"role": "user", "content": ins.strip()}
                    ],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True
                ).to(self.device)
                candis = list(
                    set(re.findall(candi_ids_part, ins.split("candidate entities:")[-1].split('relationships')[0])))
                candis.append(self.unaligned)

                if inputs['input_ids'].shape[1] > MAX_LENGTH:
                    tooLong += 1
                    if inputs['input_ids'].shape[1] > 4060:
                        rounds = 0
                        if j < len(dfs) - 1:
                            savedata.append(
                                {"target": target, "label": label, "predict": self.unaligned, "candidates": candis,
                                 'round': rounds})
                            continue

                    inputs['input_ids'] = inputs['input_ids'][:MAX_LENGTH]

                gen_kwargs = {"max_length": inputs['input_ids'].shape[1] + 100, "do_sample": True, "top_k": 1}
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]
                    predict = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # if label != '999999' or predict != '999999':
                #     print(label, 'predict', predict)
                if self.output in predict:
                    predict = predict.split("Output:")[-1]
                # print(predict)
                try:
                    predict = re.findall(r'\d+', predict)[0]
                except:
                    isNone += 1
                    predict = self.unaligned
                    # continue
                # if "\n" in predict:
                #     pre = predict.split("\n")
                #     for p in pre:
                #         if len(p) > 0:
                #             predict = p.replace(" ", "")
                # if ":" in predict:
                #     predict = predict.split(":")[0]
                # if "," in predict:
                #     predict = predict.split(",")[0]
                predicts.append(predict)

                if predict not in candis:
                    # print("label", output, "predict", predict, candis)
                    isNone += 1
                # print('label:', output, 'predict:', predict, 'pre in cad:', predict in candis, 'rounds:', j + 1)
                if predict == self.unaligned:
                    if j == len(dfs[0]) - 1:
                        no_in_candi += 1
                    else:
                        continue
                rounds += j + 1
                savedata.append(
                    {"target": target, "label": label, "predict": predict, "candidates": candis, 'round': rounds})
                if predict == label and label != self.unaligned:
                    acc += 1
                break

        get_acc(savedata)
        print("to long input:{0}, 小模型实际准确率：{1}".format(tooLong, acc_emb / len(dfs)))
        with open(save_path, 'w') as f:
            json.dump(savedata, f)
        print("path of save result:", save_path)


class Qwen_Lora:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True,
                                                       legacy=False)
        self.unaligned = '999999'
        self.device = "cuda:1"
        self.prompt = prompt1
        self.output = "Output:"

    def pad_sequences(self, sequences, pad_value, max_len=None):
        return sequences + [pad_value] * (max_len - len(sequences))

    def process_func(self, example):
        MAX_LENGTH = 2048  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
        input_ids, attention_mask, labels = [], [], []
        instruction = self.tokenizer(
            f"<|im_start|>system\n{self.prompt}<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = self.tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        # 在return前添加：
        input_ids = self.pad_sequences(input_ids, self.tokenizer.pad_token_id, MAX_LENGTH)
        attention_mask = self.pad_sequences(attention_mask, 0, MAX_LENGTH)
        labels = self.pad_sequences(labels, -100, MAX_LENGTH)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def model_lora(self, data_paths, lora_path, arg, accelerator=None):
        # 将JSON文件转换为CSV文件
        df = pd.DataFrame()
        for path in data_paths:
            df1 = pd.read_json(path)
            df = pd.concat([df, df1], axis=0)
        ds = Dataset.from_pandas(df)
        torch.cuda.empty_cache()

        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_id = ds.map(self.process_func, remove_columns=ds.column_names)

        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).to('cuda')

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
            r=8,  # Lora 秩
            lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=arg.dropout  # Dropout 比例
        )
        # 训练参数
        args = TrainingArguments(
            output_dir="./qwen_lora_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # 显存不足时补偿batch_size
            learning_rate=1e-5,
            logging_steps=100,
            num_train_epochs=arg.epoch,
            save_strategy="epoch",
            gradient_checkpointing=False,  # 可以解决参数被重复加载的问题
            ddp_find_unused_parameters=False,  # 加速训练
            logging_dir="./logs",
            remove_unused_columns=True,  # 必须为True！自动移除多余字段
            label_names=["input_ids", "attention_mask", "labels"]  # For causal LM
        )

        # 为模型添加 LoRA
        model = get_peft_model(model, config)

        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        model.print_trainable_parameters()
        # 初始化 Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_id,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, pad_to_multiple_of=8, mlm=False),
        )
        # 使用 Accelerate 准备模型和数据
        # if accelerator:
        #     model, trainer = accelerator.prepare(model, trainer)

        # 添加自定义回调函数
        # 每个批次的结果都保存
        # trainer.add_callback(SavePeftModelCallback(trainer, self.tokenizer, lora_path))

        trainer.train()
        # 只保存最后一个批次
        save_peft_model(trainer, self.tokenizer, lora_path + '_' + str(arg.epoch))

    def model_test(self, datapaths, lora_path, save_path, accelerator=None):
        print("lora path:{0}\ndata path：{1}".format(lora_path, datapaths))
        # candi_ids_part = r'(?<=\{)\d+(?=\:)'
        candi_ids_part = r'\d+(?=\:)'

        # df = pd.read_json(datapaths)
        # dfs = []
        # for path in datapaths:
        #     df = pd.read_json(path)
        #     ds = Dataset.from_pandas(df)
        #     dfs.append(ds)
        dfs = load_and_merge_json_files(datapaths)
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)

        # 加载lora权重
        model = PeftModel.from_pretrained(model, model_id=lora_path).to(self.device).eval()
        model = model.to(self.device).eval()

        acc = 0
        MAX_LENGTH = 3500
        tooLong = 0
        isNone = 0
        no_in_candi = 0
        savedata = []
        predicts = []
        # print(dfs[0].values[0])
        # for i in tqdm(range(50)):
        for i in tqdm(range(len(dfs))):
            rounds = 0
            for j in range(len(dfs[0])):
                ins, _, label, target = list(dfs[j][i].values())
                label = str(label)
                candis = list(
                    set(re.findall(candi_ids_part, ins.split("candidate entities:")[-1].split('relationships')[0])))
                candis.append('999999')

                inputs = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": ins}
                    ],
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=True,
                    return_tensors="pt",
                    return_dict=True
                ).to(self.device)
                # print(inputs)
                if inputs['input_ids'].shape[1] > MAX_LENGTH:
                    tooLong += 1
                    rounds = 0
                    if j < len(dfs) - 1:
                        savedata.append(
                            {"target": target, "label": label, "predict": self.unaligned, "candidates": candis,
                             'round': rounds})
                        break
                # gen_kwargs = {"max_length": 4096, "do_sample": False, "top_p": 0, "num_beams": 1}
                with torch.no_grad():
                    generated_ids = model.generate(
                        inputs.input_ids,
                        max_new_tokens=1024
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in
                        zip(inputs.input_ids, generated_ids)
                    ]

                    predict = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    # predict = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                print('predict is', predict)
                if self.output in predict:
                    predict = predict.split(self.output)[-1]
                    # print('predict is', predict)
                try:
                    predict = re.findall(r'\d+', predict)[0]
                except:
                    isNone += 1
                    predict = self.unaligned
                    # continue
                # print(predict, label, predict == label)
                # if "\n" in predict:
                #     pre = predict.split("\n")
                #     for p in pre:
                #         if len(p) > 0:
                #             predict = p.replace(" ", "")
                # if ":" in predict:
                #     predict = predict.split(":")[0]
                # if "," in predict:
                #     predict = predict.split(",")[0]
                predicts.append(predict)

                if predict not in candis:
                    # print("label", output, "predict", predict, candis)
                    isNone += 1
                if predict == self.unaligned:
                    if j == len(dfs) - 1:
                        no_in_candi += 1
                    else:
                        continue
                rounds += j + 1
                savedata.append(
                    {"target": target, "align": label, "predict": self.unaligned, "candidates": candis,
                     'round': rounds})
                if predict == label and label != self.unaligned:
                    acc += 1
                break

        get_acc(savedata)
        print("to long input:{0}".format(tooLong))
        with open(save_path, 'w') as f:
            json.dump(savedata, f)
        print("path of save result:", save_path)


class Qwen2vl_Lora:
    def __init__(self, model_path):
        self.model_path = model_path
        print('base model', model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True,
                                                       legacy=False)
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
        self.unaligned = '999999'
        self.device = "cuda:1"
        self.prompt = prompt1
        self.output = "Output:"

    def pad_sequences(self, sequences, pad_value, max_len=None):
        return sequences + [pad_value] * (max_len - len(sequences))

    def process_func(self, example):
        """
        将数据集进行预处理
        """
        MAX_LENGTH = 2048
        image_size = 128
        input_ids, attention_mask, labels = [], [], []
        input_content = example["instruction"]
        output_content = example["output"]
        target_img = input_content.split('\'image\': \'')[-1].split('\'}\ncandidate')[0]
        cands = input_content.split('\'image\': ')[-1].split('}\nOutput')[0]
        cands_json = cands.replace("{", "{\"").replace(": ", "\": ").replace(", ", ", \"").replace('\'', '\"')
        # print(cands_json)
        candidate_imgs = json.loads(cands_json)
        if len(target_img) < 1:
            content = [{"type": "text", "text": input_content.split(cands)[0]}]
        else:
            content = [{"type": "text", "text": input_content.split(target_img)[0]}, {
                "type": "image",
                "image": f"{target_img}",
                "resized_height": image_size,
                "resized_width": image_size,
            }, {"type": "text", "text": input_content.split(target_img)[-1][2:].split(cands)[0]}]
        for eid, img in candidate_imgs.items():
            if img == '':
                content.append({"type": "text", "text": f'{eid}: \'\''})
                continue
            content.append({"type": "text", "text": f'{eid}:'})
            content.append({
                "type": "image",
                "image": f"{img}",
                "resized_height": image_size,
                "resized_width": image_size, })
        content.append({"type": "text", "text": '}\nOutput:'})

        messages = [{"role": "assistant", "content": self.prompt},
                    {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )  # 获取文本
        image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
        # print('image_inputs', len(image_inputs))
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.tolist() for key, value in inputs.items()}  # tensor -> list,为了方便拼接
        instruction = inputs

        response = self.tokenizer(f"{output_content}", add_special_tokens=False)

        input_ids = (
                instruction["input_ids"][0] + response["input_ids"] + [self.tokenizer.pad_token_id]
        )

        attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
        labels = (
                [-100] * len(instruction["input_ids"][0])
                + response["input_ids"]
                + [self.tokenizer.pad_token_id]
        )
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        # 视觉数据处理
        if 'image_grid_thw' in inputs:
            pixel_values = np.array(inputs['pixel_values'], dtype=np.float32)
            # pixel_values = torch.tensor(inputs['pixel_values'])
            # 从 (1, h, w) 转换为 (h, w) 再调整为 (9, 3)
            image_grid_thw = np.array(inputs['image_grid_thw'], dtype=np.int64)  # (1,h,w) -> (h,w)
            # image_grid_thw = np.array(image_grid_thw, dtype=np.int64)
            if image_grid_thw.ndim == 2:  # 形状为 (h,w) 或 (N,3)
                if image_grid_thw.shape[1] != 3:  # 不是 (N,3) 格式
                    # 从 (h,w) 创建网格坐标
                    h, w = image_grid_thw.shape
                    h_pos = np.arange(h).reshape(-1, 1).repeat(w, axis=1).reshape(-1)
                    w_pos = np.arange(w).reshape(1, -1).repeat(h, axis=0).reshape(-1)
                    image_grid_thw = np.column_stack([np.ones(h * w), h_pos, w_pos])
            else:
                image_grid_thw = np.array([image_grid_thw], dtype=np.int64)
        else:
            # 默认值（确保类型和维度正确）
            pixel_values = np.zeros((image_size, image_size), dtype=np.float32)
            image_grid_thw = np.array([[1, 20, 20]], dtype=np.int64)
        # 确保所有返回的数组都是相同维度的numpy数组
        # if image_grid_thw.shape[1] != 3 or len(image_grid_thw.shape) != 2:

        # if attention_mask.ndim == 2:
        #     print('image_grid_thw', attention_mask.shape)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw
        }
        # return {
        #     'input_ids': ensure_consistent_array(input_ids, standard_shapes['input_ids']),
        #     'attention_mask': ensure_consistent_array(attention_mask, standard_shapes['attention_mask']),
        #     'labels': ensure_consistent_array(labels, standard_shapes['labels']),
        #     'pixel_values': pixel_values,
        #     'image_grid_thw': image_grid_thw
        # }

    def create_dummy_image(self, height=28, width=28, channels=1):
        """
        创建兼容PIL的默认图像（支持单通道/三通道）
        参数：
            height: 图像高度（默认28）
            width: 图像宽度（默认28）
            channels: 通道数（1-灰度，3-RGB）
        返回：
            PIL.Image对象
        """
        # 创建正确维度的数组
        if channels == 1:
            # 单通道灰度图 (H,W)
            dummy_array = np.zeros((height, width), dtype=np.uint8)
        else:
            # 三通道RGB (H,W,C)
            dummy_array = np.zeros((height, width, channels), dtype=np.uint8)

        # 添加标记（可选）
        dummy_array[:2, :] = 128  # 上边框
        dummy_array[-2:, :] = 128  # 下边框
        dummy_array[:, :2] = 128  # 左边框
        dummy_array[:, -2:] = 128  # 右边框

        return Image.fromarray(dummy_array)

    def process_input(self, example):
        """
        将数据集进行预处理
        """
        image_size = 128
        dummy_image = self.create_dummy_image(channels=3)
        input_content = example["instruction"]
        target_img = input_content.split('\'image\': \'')[-1].split('\'}\ncandidate')[0]
        cands = input_content.split('\'image\': ')[-1].split('}\nOutput')[0]
        cands_json = cands.replace("{", "{\"").replace(": ", "\": ").replace(", ", ", \"").replace('\'', '\"')
        # print(cands_json)
        candidate_imgs = json.loads(cands_json)
        if len(target_img) < 1:
            content = [{"type": "text", "text": input_content.split(cands)[0]}]
        else:
            content = [{"type": "text", "text": input_content.split(target_img)[0]}, {
                "type": "image",
                "image": f"{target_img}",
                "resized_height": image_size,
                "resized_width": image_size,
            }, {"type": "text", "text": '}' + input_content.split(target_img)[-1][2:].split(cands)[0] + '{'}]
        for eid, img in candidate_imgs.items():
            if img == '':
                content.append({"type": "text", "text": f'{eid}: \'\','})
                continue
            content.append({"type": "text", "text": f'{eid}:'})
            content.append({
                "type": "image",
                "image": f"{img}",
                "resized_height": image_size,
                "resized_width": image_size, })
        content.append({"type": "text", "text": '}\nOutput:'})

        messages = [{"role": "assistant", "content": self.prompt, },
                    {"role": "user", "content": content, }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )  # 获取文本
        image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
        # print('image_inputs', len(image_inputs))
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else dummy_image,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def model_lora(self, data_paths, lora_path, arg, accelerator=None):
        # 将JSON文件转换为CSV文件
        df = pd.DataFrame()
        for path in data_paths:
            df1 = pd.read_json(path)
            df = pd.concat([df, df1], axis=0)
            # break
        ds = Dataset.from_pandas(df)
        torch.cuda.empty_cache()

        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_id = ds.map(self.process_func, remove_columns=ds.column_names, writer_batch_size=500, load_from_cache_file=False)
        # 先加载config
        config = AutoConfig.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16
        )

        model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_path, config=config, torch_dtype=torch.bfloat16)

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            r=8,  # Lora 秩
            lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=arg.dropout  # Dropout 比例
        )
        # 训练参数
        args = TrainingArguments(
            output_dir="./qwen2vl_lora_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # 显存不足时补偿batch_size
            learning_rate=1e-5,
            num_train_epochs=arg.epoch,
            gradient_checkpointing=False,  # 可以解决参数被重复加载的问题
            ddp_find_unused_parameters=False,  # 加速训练
            logging_dir="./logs",
            save_strategy="epoch",
            remove_unused_columns=True,  # 必须为True！自动移除多余字段
        )

        # 为模型添加 LoRA
        model = get_peft_model(model, config)

        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        model.print_trainable_parameters()
        # 初始化 Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_id,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, padding=True),
        )
        # 使用 Accelerate 准备模型和数据
        if accelerator:
            model, trainer = accelerator.prepare(model, trainer)

        trainer.train()
        # 只保存最后一个批次
        save_peft_model(trainer, self.tokenizer, lora_path + '_' + str(arg.epoch))

    def model_test(self, datapaths, lora_path, save_path, accelerator=None):
        print("lora path:{0}\ndata path：{1}".format(lora_path, datapaths))
        candi_ids_part = r'(?<=\{)\d+(?=\:)'

        # df = pd.read_json(datapaths)
        dfs = []
        for path in datapaths:
            df = pd.read_json(path)
            ds = Dataset.from_pandas(df)
            dfs.append(ds)

        # 加载模型
        model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                                trust_remote_code=True)

        # 加载lora权重
        model = PeftModel.from_pretrained(model, model_id=lora_path).to(self.device).eval()
        model = model.to(self.device).eval()

        acc = 0
        MAX_LENGTH = 3500
        all_num = len(ds)
        tooLong = 0
        isNone = 0
        no_in_candi = 0
        savedata = []
        predicts = []
        # print(dfs[0].values[0])
        # for i in tqdm(range(50)):
        for i in tqdm(range(len(ds))):
            rounds = 0
            for j in range(len(dfs)):
                ins, _, label, target = list(dfs[j][i].values())
                label = str(label)
                candis = list(set(re.findall(candi_ids_part, ins.split("candidate entities:")[-1])))
                candis.append('999999')

                inputs = self.process_input(dfs[j][i]).to(self.device)
                # print(inputs)
                if inputs.input_ids.shape[1] > MAX_LENGTH:
                    tooLong += 1
                    rounds = 0
                    if j < len(dfs) - 1:
                        savedata.append(
                            {"target": target, "label": label, "predict": self.unaligned, "candidates": candis,
                             'round': rounds})
                        break
                gen_kwargs = {"do_sample": False, "top_p": 0, "num_beams": 1}
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        **gen_kwargs,
                        max_new_tokens=10
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    predict = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                print(predict)
                if self.output in predict:
                    predict = predict.split(self.output)[-1]
                try:
                    predict = re.findall(r'\d+', predict)[0]
                except:
                    isNone += 1
                    predict = self.unaligned
                    # continue
                predicts.append(predict)

                if predict not in candis:
                    # print("label", output, "predict", predict, candis)
                    isNone += 1
                if predict == self.unaligned:
                    if j == len(dfs) - 1:
                        no_in_candi += 1
                    else:
                        continue
                rounds += j + 1
                savedata.append(
                    {"target": target, "align": label, "predict": self.unaligned, "candidates": candis,
                     'round': rounds})
                if predict == label and label != self.unaligned:
                    acc += 1
                break

        get_acc(savedata)
        print("to long input:{0}".format(tooLong))
        with open(save_path, 'w') as f:
            json.dump(savedata, f)
        print("path of save result:", save_path)
