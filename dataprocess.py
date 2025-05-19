import json
import os
import random
import re
from tqdm import tqdm
import time

from sentence_transformers import SentenceTransformer
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
import torch

INSTRUCTION1 = """Input:
target entity:
{0}
candidate entities:
{1}
Output:"""

SYSTEMPROMPT4_c44MM = """Task: Performs entity alignment by determining whether aligned entities exist in candidate entities for a given target entity. When evaluating similarity, focus on information with similar attribute/relationship types, ignore different types of information, and focus on the content in the image with <image>. Two entities are aligned if the similarity of their type similar information is high. If aligned entity are found, output it's identifiers. If no aligned entity is found, 999999 is outputed. The output should only include alignment results.
Input description:
Attribute information: Format: entity_id:[(attribute type, attribute value)]
Example: 152:[('Date of birth', '1999-10-23')] indicates that entity 152 has the 'Date of birth' attribute with the value 'October 23, 1999'.
Relationship information: Format: entity_id:[(Relationship type: Number of adjacent entities)]
Example: 152:[('location contains','4')] indicates that entity 152 has four adjacent entities associated with relation type 'location contains'.
Image Format: Each image follows the format: {entity_id: <image>}, where <image> is a image, or '' if the entity has no image."""

SYSTEMPROMPT4_c45 = """Task: Perform entity alignment by identifying whether any candidate entity aligns with a given target entity.Compare attribute values only when their types are semantically similar. Relationship comparison should consider both the semantic similarity of relation types and the structural context (e.g., the number and types of connected entities). Image captions should be included in similarity evaluation only if they are not "without image".An entity is considered aligned if its attributes, relationships, and image caption exhibit high semantic similarity with those of the target.Return the entity_id of aligned entity. If none are found, return 999999.Output should include only the alignment result.
Input description:
Attribute Format: entity_id:[(attribute type, attribute value)]. Example: 152:[('Date of birth', '1999-10-23')] indicates that entity 152 has the 'Date of birth' attribute with the value 'October 23, 1999'.
Relationship Format: entity_id:[(Relationship type: Number of adjacent entities)]. Example: 152:[('location contains','4')] indicates that entity 152 has four adjacent entities associated with relation type 'location contains'.
Image caption: Each image description follows the format: {entity_id: "<caption>"}, where <caption> is a natural language sentence, or 'without image' if the entity has no image. Example: {152: 'A famous historical landmark in Paris'}
"""

device = "cuda"

# Function to clean GPU memory
def torch_gc(device):
    if torch.cuda.is_available():  # Check if CUDA is available
        with torch.cuda.device(device):  # Specify CUDA device
            torch.cuda.empty_cache()  # Clear CUDA cache
            torch.cuda.ipc_collect()  # Collect CUDA memory fragments

tokenizer = None

unalign = '999999'


# unalign = '[UNALIGNED]'


class info_fliter:
    def __init__(self, id2attr, attr_count, sent_num, allent):
        self.model = SentenceTransformer('model_path/Roberta_finetuning_semantic_similarity_stsb_multi_mt').to('cuda:0')
        self.id2attr = id2attr
        self.attr2id = [{v: k for k, v in i2a.items()} for i2a in id2attr]
        self.slen = len(id2attr[0])
        self.attr_count = attr_count
        self.sent_num = sent_num
        self.tent_num = allent - sent_num

    def attr_fliter(self, t_infos, ent_num, s_infos=None):
        t_types = {info: (info, value) for info, value in t_infos}
        if s_infos:
            s_types = {info: (info, value) for info, value in s_infos}
            types = list(s_types.keys()) + list(t_types.keys())
            slen = len(s_infos)
        else:
            types = list(t_types.keys())
            slen = 0
        tlen = len(t_infos)
        tf_score = np.zeros(tlen)

        for j, t in enumerate(t_types.keys()):
            tf = 1 / len(t_infos)
            idf = math.log(ent_num / (1 + self.attr_count[t]))
            tf_score[j] = tf * idf

        for id in range(slen, len(types)):
            if tf_score[id - slen] > 0.3:
                new_type[types[id]] = t_types[types[id]]
        return new_type


def dataset_random(filepath, label, dataset, basemodel, il):
    import os
    data = filepath.split("/")[-1].split(".json")[0]
    # if not os.path.exists("./result/" + basemodel + "/" + dataset):
    #     os.makedirs("./result/" + basemodel + "/" + dataset)
    with open(filepath, "r", encoding='utf-8') as f:
        alldata = json.load(f)
        f.close()
    random.seed(42)
    random.shuffle(alldata)
    with open(filepath, "w", encoding='utf-8') as f:
        json.dump(alldata, f, ensure_ascii=False)


def load_and_merge_json_files(file_paths):
    from collections import defaultdict
    """Read multiple JSON files and merge them grouped by target field"""
    all_data = []

    # 1. Read all file data
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)  # Merge data

    # 2. Group by target
    target_groups = defaultdict(list)
    for item in all_data:
        target_groups[item["target"]].append(item)

    # 3. Sort by target and return
    redata = [group for target, group in sorted(target_groups.items())]
    print(redata[0][0]["target"] == redata[0][1]["target"] == redata[0][2]["target"] == redata[0][3]["target"] ==
          redata[0][4]["target"])
    return [group for target, group in sorted(target_groups.items())]


def load_caption(path, dataset):
    models = ["gpt2", "blip", "Qwen"]
    processmodel = models[2]
    print(processmodel)
    allcaption = {}
    if 'FB' in dataset:
        files = ["FB15K.json"]
        if "DB" in path:
            files.append("DB15K.json")
        else:
            files.append("YAGO15K.json")
    elif 'EN' in dataset:
        if "DE" in path:
            files = ["EN_DE_id.json"]
        else:
            files = ["EN_FR_id.json"]
    for file in files:
        file = processmodel + "_captions_" + file
        with open(path + file, 'r', encoding='utf-8') as f:
            caption = json.load(f)
            f.close()
        new_cap = {}
        for k, v in caption.items():
            # new_cap[int(k)] = v
            new_cap[int(k)] = v.split('.')[0]
        allcaption = {**allcaption, **new_cap}
    return allcaption


def load_img_path(select_kg, ent2ids, captions):
    id2img_path = {}
    if type(ent2ids) is dict:
        ent2id_dict = ent2ids
    else:
        ent2id_dict = {**ent2ids[0], **ent2ids[1]}
    if 'FB' in select_kg:
        root = '/data/mmkb/images/'  # 131
        if 'DB' in select_kg:
            kgs = ['FB15K', 'DB15K']
        else:
            kgs = ['FB15K', 'YAGO15K']
        for i, kg in enumerate(kgs):
            for ent, eid in ent2ids[i].items():
                if eid in captions:
                    id2img_path[eid] = '<|vision_start|>' + root + kg + "/google_" + str(eid) + ".jpg" + '<|vision_end|>'
    elif 'EN' in select_kg:
        root = '/data/OpenEA/imgs/'
        if 'DE' in select_kg:
            kgs = ['EN_DE']
        else:
            kgs = ['EN_FR']
        for i, kg in enumerate(kgs):
            for ent, eid in ent2ids[i].items():
                if eid in captions:
                    id2img_path[eid] ='<|vision_start|>' + root + kg + "/" + str(eid) + ".png" + '<|vision_end|>'
    else:
        print('Image directory not found!')
    return id2img_path


def mutlti_instruct(result, start, end, s2t, label="eval", neg=False, pos=False):
    given_ent = []  # Source entity id
    candi_ents = []  # Candidate entity id list
    incandi = []  # label
    align_ent = []  # Target entity id
    no_candi = 0
    no_in_ills = 0

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

    neg_sum = 0
    pos_sum = 0
    random.seed(42)
    for k, v in result.items():
        if int(k) not in s2t:
            no_in_ills += 1
            continue
        ca = v[start: end]

        # Shuffle original order to improve generalization
        if label != 'eval' and pos:
            random.shuffle(ca)

        given_ent.append(int(k))
        candi_ents.append(ca)

        if s2t[int(k)] in ca:
            incandi.append(str(s2t[int(k)]))
            # Add negative samples
            if neg and label == 'train':
                # Negative sample sampling
                neg_sum += 1
                new_ca = ca.copy()
                neg_ent = random.choice(v[end:])
                given_ent.append(int(k))
                index = ca.index(s2t[int(k)])
                new_ca[index] = neg_ent
                candi_ents.append(new_ca)
                incandi.append(unalign)
        else:
            incandi.append(unalign)

            if pos and label == 'train':
                # Positive sample sampling
                pos_sum += 1
                new_ca = ca.copy()
                given_ent.append(int(k))
                neg_ent = random.choice(ca)
                index = ca.index(neg_ent)
                new_ca[index] = s2t[int(k)]
                candi_ents.append(new_ca)
                incandi.append(str(s2t[int(k)]))
        align_ent.append(s2t[int(k)])
    if neg and label == 'train':
        print("Negative sampling ratio:", neg_sum / len(result))
    if pos and label == 'train':
        print("Positive sampling ratio:", pos_sum / len(result))
    print('Number of entities:', len(given_ent))
    print("Sum of alignment entity not in candidate: {0}, number of iterations adding alignment entity: {1}".format(no_candi, no_in_ills))
    return given_ent, candi_ents, incandi


def creat_instruct(file_path, result_path, flag, num, label, dataset, basemodel, il, rate, strage):
    from dataprocess import load_data
    print('result:', result_path)
    with open(result_path, 'r') as f:
        result = json.load(f)
    for k, v in result.items():
        result[k] = v[:100]
    fliter = 'matf' in strage
    negpos = 'np' in strage
    pos = 'p' in strage
    print('Strategy parameters: attribute filtering: {0}, negative sampling: {1}, positive sampling: {2}'.format(fliter, negpos, pos))
    KGs, _, _, _ = load_data(file_path, dataset, 0.2, attr_flag=True, save=False, llm_flage=True, fliter=fliter)
    if os.path.exists(file_path + dataset):
        caption = load_caption(file_path + dataset + '/', dataset)
    else:
        if 'EN' in dataset:
            caption = load_caption(file_path + 'imgs/', [dataset.split('_15K')[0]])
        else:
            print('Caption directory not found')
            caption = [{}, {}]
    entities = KGs["node"]  # id2attr
    old_ent_rels = KGs["rel"]  # id2rel
    ills = KGs["ills"]  # alignment pairs:[(sid,tid)]
    # fliter = info_fliter(KGs['id2attr'], KGs['attr_count'], KGs['ent_num'], len(entities))
    del KGs
    ent_rels = {}
    for ent, rels in old_ent_rels.items():
        for rel, neigs in rels.items():
            if ent not in ent_rels:
                # ent_rels[ent] = {rel: list(neigs)}
                ent_rels[ent] = [(rel, str(len(neigs)))]
            else:
                ent_rels[ent].append((rel, str(len(neigs))))

    s2t = {}
    t2s = {}
    for i, j in ills:
        s2t[i] = j
        t2s[j] = i

    target_id = []
    align_res = []
    candi_ids_part = r'\d+(?=\:)'
    round = 2 if label == 'train' else 5
    start = 0
    end = num
    save_paths = []
    # given_ent, candi_ents, incandi = mutlti_instruct(result, start, end, s2t, label, neg=True)
    given_ent, candi_ents, incandi = mutlti_instruct(result, start, end, s2t, label, neg=negpos)
    if flag:
        return given_ent, candi_ents, ent_rels, incandi
    else:
        for number in range(round):
            print("start:", start, "end:", end)
            error = 0
            instructs = []
            # parallel_process_entities(given_ent, entities, ent_rels, caption, candi_ents, incandi, label)
            # """
            for i in tqdm(range(len(given_ent))):
                given = {}
                attr = {}
                rel = {}
                cap = {}
                # info = {"relatinship": {given_ent[i]: ent_rels[given_ent[i]]}, "attribution": {given_ent[i]: entities[given_ent[i]]["attrs"]}}
                ent_caption = caption.get(given_ent[i], "without image")

                info = {"attribute": list(entities[given_ent[i]]["attrs"]), "relationship": ent_rels[given_ent[i]],
                        "caption": ent_caption}
                # given[given_ent[i]] = info
                for j in candi_ents[i]:
                    attr[j] = list(entities[j]["attrs"])
                    rel[j] = ent_rels.get(j, [])
                    cap[j] = caption.get(j, "without image")
                candi = {"attribute": attr, "relationship": rel, "caption": cap}
                instruct = {"instruction": SYSTEMPROMPT4_c44 + INSTRUCTION1.format(info, candi),
                                "input": "",
                                "output": str(incandi[i]),
                                "target": str(given_ent[i]),
                                }
                candis = list(
                    set(re.findall(candi_ids_part,
                                   instruct['instruction'].split("candidate entities:")[-1].split('relationships')[0])))
                if str(instruct['output']) not in candis and str(instruct['output']) != unalign:
                    print(instruct['output'], candis)
                    error += 1
                instructs.append(instruct)
              
            print('Error data count:', error, 'Total data count:', len(given_ent))
            save_dir = "result/{0}/{1}/{2}_{3}_{4}".format(basemodel, il, dataset.replace('/norm', '')[:-3], rate,
                                                           strage)
            save_path = save_dir + "/instruct_zero_{0}-{1}_{2}.json".format(start, end, label)
            save_paths.append(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print(save_path)
            with open(save_path, 'w', encoding="utf-8") as f:
                json.dump(instructs, f, ensure_ascii=False)

            start = end
            end += num
            if number >= round - 1:
                break
            if label != 'train':
                given_ent, candi_ents, incandi = mutlti_instruct(result, start, end, s2t)
            else:
                # given_ent, candi_ents, incandi = mutlti_instruct(result, start, end, s2t, label, pos=True)
                given_ent, candi_ents, incandi = mutlti_instruct(result, start, end, s2t, label, pos=negpos)


def creat_MM_instruct(file_path, result_path, flag, num, label, dataset, basemodel, il, rate, strage):
    from dataprocess import load_data
    print('result:', result_path)
    # result = {}
    with open(result_path, 'r') as f:
        result = json.load(f)
    for k, v in result.items():
        result[k] = v[:100]
    KGs, _, _, _ = load_data(file_path, dataset, 0.2, attr_flag=True, save=False, llm_flage=True)
    if os.path.exists(file_path + dataset):
        caption = load_caption(file_path + dataset + '/', dataset)
    else:
        if 'EN' in dataset:
            caption = load_caption(file_path + 'imgs/', [dataset.split('_15K')[0]])
        else:
            caption = [{}, {}]
    id2img_path = load_img_path(dataset, KGs["ent2id"], caption)
    entities = KGs["node"]  # id2attr
    old_ent_rels = KGs["rel"]  # id2rel
    ills = KGs["ills"]  # alignment pairs:[(sid,tid)]
    # fliter = info_fliter(KGs['id2attr'], KGs['attr_count'], KGs['ent_num'], len(entities))
    del KGs
    ent_rels = {}
    for ent, rels in old_ent_rels.items():
        for rel, neigs in rels.items():
            if ent not in ent_rels:
                # ent_rels[ent] = {rel: list(neigs)}
                ent_rels[ent] = [(rel, str(len(neigs)))]
            else:
                ent_rels[ent].append((rel, str(len(neigs))))
    s2t = {}
    t2s = {}
    for i, j in ills:
        s2t[i] = j
        t2s[j] = i

    target_id = []
    align_res = []

    round = 2 if label == 'train' else 5
    start = 0
    end = 10
    given_ent, candi_ents, incandi = mutlti_instruct(result, start, end, s2t, label, neg=True)
    if flag:
        return given_ent, candi_ents, ent_rels, incandi
    else:
        for number in range(round):
            print("start:", start, "end:", end)
            instructs = []
            for i in tqdm(range(len(given_ent))):
                given = {}
                attr = {}
                rel = {}
                cap = {}
                info = {'text': {"attributes": list(entities[given_ent[i]]["attrs"]),
                                 "relationships": ent_rels[given_ent[i]]}, 'image': id2img_path.get(i, "")}
                for j in candi_ents[i]:
                    attr[j] = list(entities[j]["attrs"])
                    rel[j] = ent_rels.get(j, [])
                    cap[j] = id2img_path.get(j, "")
                candi = {'test': {"attributes": attr, "relationships": rel}, 'image': cap}

                instruct = {"instruction": SYSTEMPROMPT4_c44MM + INSTRUCTION1.format(info, candi),
                            "input": "",
                            "output": incandi[i],
                            "target": given_ent[i]}
                instructs.append(instruct)
            # """
            save_dir = "result/{0}/{1}/{2}_{3}_MM_{4}".format(basemodel, il, dataset.replace('/norm', '')[:-3], rate,
                                                              strage)
            save_path = save_dir + "/instruct_zero_{0}-{1}_{2}.json".format(start, end, label)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print(save_path)
            with open(save_path, 'w', encoding="utf-8") as f:
                json.dump(instructs, f, ensure_ascii=False)
            # start += num * (number + 1)
            # end = start + num * (number + 2)
            start = end
            end += 10
            if number >= round - 1:
                break
            if label != 'train':
                given_ent, candi_ents, incandi = mutlti_instruct(result, start, end, s2t)
            else:
                given_ent, candi_ents, incandi = mutlti_instruct(result, start, end, s2t, label, pos=True)


def result_process(file_path, result_path):
    """
    :param file_path:
    :param result_path: reslut_dict. {entid: [candidate_ent_index]}
    :return:
    """
    from transformers import AutoModelForCausalLM
    glm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True)
    pattern = r'(?<=$$|\{)\d+(?=$$|\})'
    given_ent, candi_ents, ent_rels, incandi = creat_instruct(file_path, result_path, flag=1, num=5)
    align_res = []
    print("start LLM")
    for i in range(len(given_ent)):
        given = {}
        candidates = []
        candi = []
        given[given_ent[i]] = ent_rels[given_ent[i]]
        for j in candi_ents[i]:
            candi.append({j: ent_rels[j]})
        # candidates.append({given_ent[i]: candi})
        # prompt = {"input": INSTRUCTION.format(given, candi, candi_ents[i], list(given.keys())),"system": SYSTEMPROMPT}
        prompt = {"input": INSTRUCTION.format(given, candi), "system": SYSTEMPROMPT}
        start = time.time()
        output = glm(prompt)
        # print(time.time()-start)
        print(output)
        print("candi:{}".format(incandi[i]))
        if output != "50000":
            if "[" in output or "{" in output:
                try:
                    output = re.findall(pattern, output)[0]
                except:
                    output = "999999"
            align_res.append(int(output))
        else:
            align_res.append(50000)
        if i == 20 - 1:
            break
    acc_sum = 0  # Number of correct entities filtered by LLM
    pre_sum = 0  # Number of correct entities in LLM input candidate set
    toolong = 0
    for i in range(len(align_res)):
        if align_res[i] == incandi[i]:
            acc_sum += 1
        if incandi[i] < 999999:
            pre_sum += 1
        if align_res[i] == 40000:
            toolong += 1
    print("Accuracy of LMM: {0}".format(acc_sum / pre_sum))
    print("Number of inputs that were too long: {0}".format(toolong))
    print("end!")


if __name__ == "__main__":
    basemodel = "PMF"
    il = "nil"
    split = ''
    for dataset in ["FBYG", "FBDB"]:
        print("basemodel is:", basemodel, "dataset is:", dataset, "il:", il)
        if 'FB' in dataset:
            root = 'mmkb'
            dataset = dataset + '15K'
            dataset1 = dataset
        elif 'DBP' in dataset:
            root = 'DBP15K'
            dataset = dataset.split('K_')[-1]
            dataset1 = root + '_' + dataset.split('K_')[-1]
        else:
            root = 'OpenEA'
            dataset = dataset + '_15K_V2'
            dataset1 = dataset
            split = '/norm_'
        if basemodel == "PMF":
            # dataset = root + '_' + dataset
            dataset1 = basemodel + "_" + dataset1
        else:
            dataset1 = dataset
        for rate in ['0.2', '0.5', '0.8']:
            for label in ["train", "eval"]:
                if label == "train":
                    strage = "matf_l_np"
                else:
                    strage = "matf"
                label1 = label
                # creat_MM_instruct(file_path=f"/data/{root}/",
                creat_instruct(file_path=f"/data/{root}/",
                                   result_path="./result/{0}/{1}/{2}_{3}_{1}_left_{4}.json".format(basemodel, il,
                                                                                                   dataset1,
                                                                                                   split[1:] + rate,
                                                                                                   label1),
                                   flag=0, num=10, label=label, dataset=dataset + split[:-1], basemodel=basemodel,
                                   il=il, rate=rate, strage=strage)
