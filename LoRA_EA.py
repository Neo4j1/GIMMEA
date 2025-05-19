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
from torch.nn import functional as F  # Key import
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, \
    TrainerCallback, Qwen2VLForConditionalGeneration, AutoConfig
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, \
    AutoProcessor

# Get number of available GPUs
gpu_count = torch.cuda.device_count()
print(f"Number of available GPUs: {gpu_count}")

prompt = "You're an expert in knowledge graph alignment.The alignment of the target entity can be analyzed based on the degree of similarity of information between the candidate entity and the target entity."
prompt1 = "You're an expert in knowledge graph alignment."
prompt2 = "Suppose you are an expert in knowledge graph alignment."


def save_peft_model(trainer, tokenizer, lora_path):
    peft_model_id = lora_path
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    print("Model saved successfully!")


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
    print('Small model accuracy:', 1 - noin / all)
    # print('Fine-tuned model accuracy:', acc / all)
    print("Alignment accuracy:", acc / all)
    print("Average round:", rs / all)
    print("Number of aligned entities:", acc)
    # print('Average inference rounds:', rs / all)


class SavePeftModelCallback(TrainerCallback):
    def __init__(self, trainer, tokenizer, base_save_path):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.base_save_path = base_save_path  # Base save path

    def on_epoch_end(self, args, state, control, **kwargs):
        # Dynamically generate save path at end of each epoch, e.g.: ./save_model_epoch_1, ./save_model_epoch_2
        save_path = f"{self.base_save_path}_{round(state.epoch)}"
        print('Model save path:', save_path)
        save_peft_model(self.trainer, self.tokenizer, save_path)
        return control


def vectorized_find_sequence(b_tensor, a_list, eqal=True):
    if isinstance(a_list, torch.Tensor):
        # Option 1: Clone existing tensor
        a_tensor = a_list.clone().detach().to(b_tensor.device)
    else:
        # Option 2: Create from Python data
        a_tensor = torch.tensor(a_list, device=b_tensor.device)
    a_len = len(a_list)

    # Generate all sliding windows [b_len - a_len + 1, a_len]
    windows = b_tensor.unfold(-1, a_len, 1)

    # Compare all windows [..., num_windows]
    match_mask = (windows == a_tensor).all(-1)

    # Generate final mask
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
        self.unaligned_token_id = unaligned_token_id  # Token ID for unaligned identifier
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.current_epoch = 0  # Track current epoch

    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. Forward pass (disable cache to save memory)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False  # Must be disabled!
        )
        logits = outputs.logits
        
        if hasattr(self, 'state') and self.state is not None:
            self.current_epoch = self.state.epoch
            
        # 2. Batch-wise loss calculation (key improvement)
        # Modified: Select loss calculation method based on epoch
        if self.current_epoch <= 2:  # First 2 epochs use only CE
            loss = self.ce_loss(logits, inputs["labels"])
        else:  # From 3rd epoch add custom loss
            loss = self._compute_memory_efficient_loss(
                logits=logits,
                labels=inputs["labels"],
                unaligned_token_id=self.unaligned_token_id
            )

        return (loss, outputs) if return_outputs else loss

    def _compute_memory_efficient_loss(self, logits, labels, unaligned_token_id):
        # Initialize total loss
        total_loss = torch.tensor(0.0, device=logits.device)

        # 3. Sample-wise calculation (avoid memory explosion from matrix operations)
        for i in range(logits.shape[0]):
            # Extract current sample's logits and labels
            sample_logits = logits[i, :, :]  # [seq_len, vocab_size]
            sample_labels = labels[i, :]  # [seq_len]

            # Calculate standard cross entropy (only non-padding positions)
            ce_loss = F.cross_entropy(
                sample_logits.view(-1, sample_logits.size(-1)),
                sample_labels.view(-1),
                ignore_index=-100
            )

            # Modified: Gradually introduce custom loss after 3rd epoch
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

                # Modified: Progressive weighting (increase weight with epochs)
                custom_loss_weight = min(1.0, (self.current_epoch - 2) / 3.0)  # From 0.33→1.0 between 3rd-5th epoch
                total_loss += ce_loss + custom_loss_weight * (
                        0.5 * (align_penalty + unalign_penalty) - 1.0 * reward
                )
                print('CE loss:', ce_loss.item(), 'align_penalty', align_penalty.item(), 'unalign_penalty',
                      unalign_penalty.item(), 'reward', reward.item())
            else:
                total_loss += ce_loss

        print('Average loss:', total_loss / logits.shape[0])

        return total_loss / logits.shape[0]  # Return batch average loss


def model_param(model):
    names = []
    for name, param in model.named_parameters():
        if name not in names:
            names.append(name)
        else:
            print(name)


def load_and_merge_json_files(file_paths):
    """Read multiple JSON files and group by target field"""
    all_data = []

    # 1. Read all file data
    for path in file_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            all_data.extend(data)  # Merge data

    # 2. Group by target
    target_groups = defaultdict(list)
    for item in all_data:
        target_groups[item["target"]].append(item)

    # 3. Return sorted by target
    redata = [group for target, group in sorted(target_groups.items())]
    print(redata[0][0]["target"] == redata[0][1]["target"] == redata[0][2]["target"] == redata[0][3]["target"] == redata[0][4]["target"])
    return [group for target, group in sorted(target_groups.items())]


class GLM_Lora:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
        self.unaligned = '999999'
        self.train = True
        self.device = "cuda:1"
        self.prompt = prompt1
        self.output = "Output:"

    def process_func(self, example):
        MAX_LENGTH = 2048
        input_ids, attention_mask, labels = [], [], []
        instruction = self.tokenizer((f"[gMASK]<sop><|system|>\n{self.prompt}<|user|>\n"
                                      f"{example['instruction'] + example['input']}<|assistant|>\n"
                                      ).strip(),
                                     add_special_tokens=False)

        response = self.tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # Set to 1 since we also care about eos token
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:  # Truncate if needed
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def model_lora(self, data_paths, lora_path, arg, accelerator=None):
        # Convert JSON files to CSV
        dfs = []
        for path in data_paths:
            df = pd.read_json(path)
            dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle data using sample
        ds = Dataset.from_pandas(df)
        torch.cuda.empty_cache()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_id = ds.map(self.process_func, remove_columns=ds.column_names)

        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
        model_param(model)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["query_key_value"],
            inference_mode=False,  # Training mode
            r=8,  # Lora rank
            lora_alpha=32,  # Lora alpha, see Lora principles
            lora_dropout=arg.dropout  # Dropout ratio
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
            ddp_find_unused_parameters=False  # Speed up training
        )

        model = get_peft_model(model, config)

        model.enable_input_require_grads()
        model.config.use_cache = False

        model.print_trainable_parameters()

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_id,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
        )

        if accelerator:
            # Prepare model and data with Accelerate
            model, trainer = accelerator.prepare(model, trainer)

        trainer.train()
        # Save only the last batch
        save_peft_model(trainer, self.tokenizer, lora_path + '_' + str(arg.epoch))

    def model_test(self, datapaths, lora_path, save_path, accelerator=None):
        self.train = False
        torch.cuda.set_device(self.device)

        print("lora path:{0}\ndata path：{1}".format(lora_path, datapaths))
        candi_ids_part = r'\d+(?=\:)'

        dfs = load_and_merge_json_files(datapaths)
        print(len(dfs[0]), dfs[0][0]['target'] == dfs[0][1]['target'] == dfs[0][2]['target'] == dfs[0][3]['target'] == dfs[0][4]['target'])
        # Load model
        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)

        # Load lora weights
        model = PeftModel.from_pretrained(model, model_id=lora_path).to(self.device).eval()

        acc = 0
        MAX_LENGTH = 4000
        tooLong = 0
        isNone = 0
        no_in_candi = 0
        acc_emb = 0
        savedata = []
        predicts = []
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
                if self.output in predict:
                    predict = predict.split("Output:")[-1]
                try:
                    predict = re.findall(r'\d+', predict)[0]
                except:
                    isNone += 1
                    predict = self.unaligned
                predicts.append(predict)

                if predict not in candis:
                    isNone += 1
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
        print("Too long inputs:{0}, Small model actual accuracy: {1}".format(tooLong, acc_emb / len(dfs)))
        with open(save_path, 'w') as f:
            json.dump(savedata, f)
        print("Result save path:", save_path)


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
        MAX_LENGTH = 2048  # Qwen tokenizer may split a Chinese character into multiple tokens, so need to increase max length
        input_ids, attention_mask, labels = [], [], []
        instruction = self.tokenizer(
            f"<|im_start|>system\n{self.prompt}<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False)  # add_special_tokens doesn't add special tokens at beginning
        response = self.tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # Set to 1 since we also care about eos token
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:  # Truncate if needed
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        # Add before return:
        input_ids = self.pad_sequences(input_ids, self.tokenizer.pad_token_id, MAX_LENGTH)
        attention_mask = self.pad_sequences(attention_mask, 0, MAX_LENGTH)
        labels = self.pad_sequences(labels, -100, MAX_LENGTH)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def model_lora(self, data_paths, lora_path, arg, accelerator=None):
        # Convert JSON files to CSV
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
            r=8,  # Lora rank
            lora_alpha=32,  # Lora alpha, see Lora principles
            lora_dropout=arg.dropout  # Dropout ratio
        )
        # Training parameters
        args = TrainingArguments(
            output_dir="./qwen_lora_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # Compensate for insufficient memory
            learning_rate=1e-5,
            logging_steps=100,
            num_train_epochs=arg.epoch,
            save_strategy="epoch",
            gradient_checkpointing=False,  # Can solve parameter reloading issues
            ddp_find_unused_parameters=False,  # Speed up training
            logging_dir="./logs",
            remove_unused_columns=True,  # Must be True! Automatically removes extra fields
            label_names=["input_ids", "attention_mask", "labels"]  # For causal LM
        )

        # Add LoRA to model
        model = get_peft_model(model, config)

        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        model.print_trainable_parameters()
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_id,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, pad_to_multiple_of=8, mlm=False),
        )

        trainer.train()
        # Save only the last batch
        save_peft_model(trainer, self.tokenizer, lora_path + '_' + str(arg.epoch))

    def model_test(self, datapaths, lora_path, save_path, accelerator=None):
        print("lora path:{0}\ndata path：{1}".format(lora_path, datapaths))
        candi_ids_part = r'\d+(?=\:)'

        dfs = load_and_merge_json_files(datapaths)
        # Load model
        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)

        # Load lora weights
        model = PeftModel.from_pretrained(model, model_id=lora_path).to(self.device).eval()
        model = model.to(self.device).eval()

        acc = 0
        MAX_LENGTH = 3500
        tooLong = 0
        isNone = 0
        no_in_candi = 0
        savedata = []
        predicts = []
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
                if inputs['input_ids'].shape[1] > MAX_LENGTH:
                    tooLong += 1
                    rounds = 0
                    if j < len(dfs) - 1:
                        savedata.append(
                            {"target": target, "label": label, "predict": self.unaligned, "candidates": candis,
                             'round': rounds})
                        break
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

                print('predict is', predict)
                if self.output in predict:
                    predict = predict.split(self.output)[-1]
                try:
                    predict = re.findall(r'\d+', predict)[0]
                except:
                    isNone += 1
                    predict = self.unaligned
                predicts.append(predict)

                if predict not in candis:
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
        print("Too long inputs:{0}".format(tooLong))
        with open(save_path, 'w') as f:
            json.dump(savedata, f)
        print("Result save path:", save_path)


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
        Preprocess dataset
        """
        MAX_LENGTH = 2048
        image_size = 128
        input_ids, attention_mask, labels = [], [], []
        input_content = example["instruction"]
        output_content = example["output"]
        target_img = input_content.split('\'image\': \'')[-1].split('\'}\ncandidate')[0]
        cands = input_content.split('\'image\': ')[-1].split('}\nOutput')[0]
        cands_json = cands.replace("{", "{\"").replace(": ", "\": ").replace(", ", ", \"").replace('\'', '\"')
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
        )  # Get text
        image_inputs, video_inputs = process_vision_info(messages)  # Get image data (preprocessed)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.tolist() for key, value in inputs.items()}  # tensor -> list, for easy concatenation
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
        if len(input_ids) > MAX_LENGTH:  # Truncate if needed
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        # Visual data processing
        if 'image_grid_thw' in inputs:
            pixel_values = np.array(inputs['pixel_values'], dtype=np.float32)
            image_grid_thw = np.array(inputs['image_grid_thw'], dtype=np.int64)
            if image_grid_thw.ndim == 2:
                if image_grid_thw.shape[1] != 3:
                    h, w = image_grid_thw.shape
                    h_pos = np.arange(h).reshape(-1, 1).repeat(w, axis=1).reshape(-1)
                    w_pos = np.arange(w).reshape(1, -1).repeat(h, axis=0).reshape(-1)
                    image_grid_thw = np.column_stack([np.ones(h * w), h_pos, w_pos])
            else:
                image_grid_thw = np.array([image_grid_thw], dtype=np.int64)
        else:
            pixel_values = np.zeros((image_size, image_size), dtype=np.float32)
            image_grid_thw = np.array([[1, 20, 20]], dtype=np.int64)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw
        }

    def create_dummy_image(self, height=28, width=28, channels=1):
        """
        Create PIL-compatible default image (supports single/three channels)
        Parameters:
            height: Image height (default 28)
            width: Image width (default 28)
            channels: Number of channels (1-grayscale, 3-RGB)
        Returns:
            PIL.Image object
        """
        if channels == 1:
            dummy_array = np.zeros((height, width), dtype=np.uint8)
        else:
            dummy_array = np.zeros((height, width, channels), dtype=np.uint8)

        dummy_array[:2, :] = 128  # Top border
        dummy_array[-2:, :] = 128  # Bottom border
        dummy_array[:, :2] = 128  # Left border
        dummy_array[:, -2:] = 128  # Right border

        return Image.fromarray(dummy_array)

    def process_input(self, example):
        """
        Preprocess input data
        """
        image_size = 128
        dummy_image = self.create_dummy_image(channels=3)
        input_content = example["instruction"]
        target_img = input_content.split('\'image\': \'')[-1].split('\'}\ncandidate')[0]
        cands = input_content.split('\'image\': ')[-1].split('}\nOutput')[0]
        cands_json = cands.replace("{", "{\"").replace(": ", "\": ").replace(", ", ", \"").replace('\'', '\"')
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
        )  # Get text
        image_inputs, video_inputs = process_vision_info(messages)  # Get image data (preprocessed)
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else dummy_image,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def model_lora(self, data_paths, lora_path, arg, accelerator=None):
        # Convert JSON files to CSV
        df = pd.DataFrame()
        for path in data_paths:
            df1 = pd.read_json(path)
            df = pd.concat([df, df1], axis=0)
        ds = Dataset.from_pandas(df)
        torch.cuda.empty_cache()

        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_id = ds.map(self.process_func, remove_columns=ds.column_names, writer_batch_size=500, load_from_cache_file=False)
        # First load config
        config = AutoConfig.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16
        )

        model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_path, config=config, torch_dtype=torch.bfloat16)

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            r=8,  # Lora rank
            lora_alpha=32,  # Lora alpha, see Lora principles
            lora_dropout=arg.dropout  # Dropout ratio
        )
        # Training parameters
        args = TrainingArguments(
            output_dir="./qwen2vl_lora_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # Compensate for insufficient memory
            learning_rate=1e-5,
            num_train_epochs=arg.epoch,
            gradient_checkpointing=False,  # Can solve parameter reloading issues
            ddp_find_unused_parameters=False,  # Speed up training
            logging_dir="./logs",
            save_strategy="epoch",
            remove_unused_columns
