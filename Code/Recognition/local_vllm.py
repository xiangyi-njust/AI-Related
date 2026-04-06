import torch
from vllm import LLM, SamplingParams
import datetime
from tqdm import tqdm
import logging
import os
import pandas as pd
import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
system_data_dir = os.environ.get("DATA")


"""
    如果是小模型，直接使用bfloat16
    如果是中等模型
        1) 使用8bit, quantization='bitsandbytes'  dtype='bfloat16'
        2) 使用4bit，模型版本切换为AWQ版本，quantization='awq' dtype='float16'
"""
settings = [
    # 单卡A40
    {
        'model_id': f"{system_data_dir}/datasets/llama3_weight/Meta-Llama-3.1-8B-Instruct",
        'dtype': "bfloat16",
        'quantization': None,
        'tar_file': "../data/extract_by_llm/llama_8B_v3/compare_gpt.json",
        'tensor_parallel_size':1,
    },
    {
        'model_id': f"{system_data_dir}/datasets/llama3_weight/Qwen2.5-32B-Instruct-AWQ",
        'dtype': "float16",
        'quantization': 'awq',
        'tar_file': "../data/extract_by_llm/Qwen_32B_AWQ_v3/compare_gpt.json",
        'tensor_parallel_size':1,
    },
    # 双卡A40
    {
        'model_id': f"{system_data_dir}/datasets/llama3_weight/Qwen2.5-32B-Instruct",
        'dtype': "bfloat16",
        'quantization': None,
        'tensor_parallel_size':2,
        'tar_file': "../data/extract_by_llm/Qwen_32B_v3/compare_gpt.json",
        'distributed_executor_backend':'mp',
        'enforce_eager':True,
    },
    {
        'model_id': f"{system_data_dir}/datasets/llama3_weight/Qwen2.5-72B-Instruct-AWQ",
        'dtype': "float16",
        'quantization': 'awq',
        'tensor_parallel_size':2,
        'tar_file': "../data/extract_by_llm/Qwen_72B_AWQ_v3/compare_gpt.json",
        'distributed_executor_backend':'mp',
        'enforce_eager':True,
    },
    {
        'model_id': f"{system_data_dir}/datasets/llama3_weight/Llama-3.3-70B-Instruct-AWQ",
        'dtype': "float16",
        'quantization': 'awq',
        'tensor_parallel_size':2,
        'tar_file': "../data/extract_by_llm/llama_70B_AWQ_v3/compare_gpt.json",
        'distributed_executor_backend':'mp',
        'enforce_eager':True,
    },
]

setting = settings[3]
llm = LLM(
    model=setting['model_id'],
    quantization=setting['quantization'],
    dtype=setting['dtype'],
    tensor_parallel_size=setting['tensor_parallel_size'],  # 如果有多GPU可以增加
    gpu_memory_utilization=0.85,  # GPU内存利用率
    distributed_executor_backend=setting.get('distributed_executor_backend', None),
    enforce_eager=setting.get('enforce_eager', False),
    max_model_len=setting.get('max_model_len', 10240),
)
tokenizer = llm.get_tokenizer()


class Logger(object):
    def __init__(self, filename, level='info'):
        level = logging.INFO if level == 'info' else logging.DEBUG
        self.logger = logging.getLogger(filename)
        self.logger.propagate = False
        self.logger.setLevel(level)

        th = logging.FileHandler(filename, 'a')

        self.logger.addHandler(th)


log = Logger(f"logs/output.log")


system_prompt = """
You are a research assistant that analyzes abstracts from academic research papers in the field of management.
"""

user_prompt = """
Determine whether the abstract occur any concept, technique, model, algorithm or tool related to artificial intelligence.
Such items include (but are not limited to):
- General AI-related terms (e.g., artificial intelligence, machine learning, natural language processing, computer vision, computational linguistics)
- General AI algorithms(e.g., deep learning models, machine learning methods, reinforcement learning algorithms, cluster algorithms)
- Specific AI algorithms (e.g., BERT, LSTM, CNN)
- Classical machine learning methods (e.g., decision tree, support vector machine, k-means, k-nearest-neighbors)
- AI research tasks (e.g., text-based analysis, image-based analysis, cluster analysis, sentiment analysis)
- AI applications (e.g., recommendation system, chatbot)
- Statistical learning methods（e.g., Bayes model, Monte Carlo simulations, Markov model, Lasso, ridge regression）
- Any methods potentially originating from the AI field (e.g., expert systems, agent-based simulation)

(1) If any of these terms appear in the abstract, return "Yes"; otherwise, return "No".
(2) Generic mentions such as "artificial intelligence" or "machine learning" should also lead to a "Yes".
(3) Do NOT limit yourself to a predefined list. Use your internal knowledge.
(4) Look for phrases like "data-driven approach," "predictive modeling," or "algorithmic classification" even if specific algorithm names are missing.
(5) If you are unsure whether a method counts as "AI" or "Statistics", in this specific task, return "Yes".

Note: Just return "Yes" or "No", do not include any additional words.

Example:
{
    Input: "Based on multi-criteria decision analysis, this study applies particle swarm optimization to enhance supply chain inventory strategy."
    Output: "Yes"
}

Now analyze the following abstract\n:
"""


def format_conversation(abstract):

    """格式化对话为chat模板"""
    if len(abstract) > 12000:
        abstract = abstract[:12000] + "..."

    new_conversation = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"{user_prompt}Input:{abstract}\nOutput:"}
    ]
   
    formatted_conversation = tokenizer.apply_chat_template(
        new_conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    return formatted_conversation


def infer(conversations):
    global llm

    log.logger.info(f"{datetime.datetime.now().time()}: start formatting conversations")

    # 格式化所有对话
    formatted_texts = [format_conversation(conv) for conv in conversations]
    log.logger.info(f"{datetime.datetime.now().time()}: start inference with vLLM")

    # 设置采样参数
    sampling_params = SamplingParams(
        max_tokens=2,
        temperature=0,  # do_sample=False 等价于 temperature=0
        top_p=1.0,
        skip_special_tokens=True
    )

    # vLLM批量推理 - 自动处理所有batching
    outputs = llm.generate(formatted_texts, sampling_params)

    # 提取生成的文本
    tags = [output.outputs[0].text.strip() for output in outputs]
    log.logger.info(f"{datetime.datetime.now().time()}: finish inference")
    return tags


def test_performance():
    # 基于chatgpt得到的结果
    with open("../data/from_old_project/first_round_res.json", 'r') as f:
        datas = json.load(f)
        ai_dois = list(datas.keys())
        ai_dois = ["https://doi.org/" + doi for doi in ai_dois]
    
    # 基于chatgpt得到的结果--补充的ai_dois
    supply_ai_dois = []
    with open("../data/gpt_predict/round_1_supply/pred_by_optim_v1_ai_dois.txt", 'r') as f:
        for line in f.readlines():
            supply_ai_dois.append("https://doi.org/" + line.strip())
    ai_dois.extend(supply_ai_dois)
    print(len(ai_dois))

    ranking = '4*'
    files = os.listdir(f"../data/journal/{ranking}/have_abs")
    files.sort()

    cnt = 0 
    paper_datas = {}
    for file in tqdm(files):
        ori_file = os.path.join(f"../data/journal/{ranking}/have_abs/", file)
        df = pd.read_csv(ori_file, keep_default_na=False)
        df = df[df['doi'].isin(ai_dois)]
        titles = df['title'].tolist()
        abstracts = df['abstract'].tolist()
        if len(titles) == 0:
            continue
        else:
            cnt += len(titles)

        texts = [title + '\n' + abstract for title, abstract in zip(titles, abstracts)]
        tags = infer(texts)
        df['tag'] = tags
        new_paper_dats = df.set_index('id').to_dict("index")
        paper_datas.update(new_paper_dats)
    
    print(cnt)

    with open(setting['tar_file'], 'w') as f:
        json.dump(paper_datas, f, indent=4)


def predict(ranking="4*"):
    files = os.listdir(f"../data/journal/{ranking}/have_abs")
    files.sort()
    for file in tqdm(files):
        ori_file = os.path.join(f"../data/journal/{ranking}/have_abs/", file)
        tar_file = os.path.join(f"../data/extract_by_llm/llama_70B_AWQ_v3/{ranking}", file.replace(".csv", ".json"))

        paper_datas = {}
        finish_ids = []
        if os.path.exists(tar_file):
            with open(tar_file, 'r') as f:
                paper_datas = json.load(f)
                finish_ids = list(paper_datas.keys())

        df = pd.read_csv(ori_file, keep_default_na=False)
        df = df[~df['id'].isin(finish_ids)]
        print(df.shape)
        titles = df['title'].tolist()
        abstracts = df['abstract'].tolist()
        if len(titles) == 0:
            continue
        texts = [title + '\n' + abstract for title, abstract in zip(titles, abstracts)]
        
        tags = infer(texts)
        df['tag'] = tags

        new_paper_dats = df.set_index('id').to_dict("index")
        paper_datas.update(new_paper_dats)
        with open(tar_file, 'w') as f:
            json.dump(paper_datas, f, indent=4)


def main():
    test_performance()

    # rankings = ["4*", "4", "3", "2", "1"]
    # for ranking in rankings:
    #     predict(ranking)


if __name__ == '__main__':
    main()