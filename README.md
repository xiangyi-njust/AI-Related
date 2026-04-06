# AI-Related Research Project

This repository contains research code and data for analyzing AI-related academic papers in the field of management research.

## Project Overview

This project focuses on identifying and analyzing academic papers that utilize artificial intelligence (AI) methods in management research. The work involves:

- **Journal Data Collection**: Gathering publication data from various management journals
- **AI Recognition**: Using large language models (LLMs) to identify AI-related papers from abstracts
- **Validation**: Comparing model predictions against ground truth labels
- **Journal Ranking**: Working with ABS (Association of Business Schools) Journal Rankings

## Repository Structure

```
AI-Related/
├── Code/
│   ├── Evolution/          # (Empty directory - reserved for future work)
│   └── Recognition/
│       ├── local_vllm.py          # Main inference script for AI paper classification
│       └── validate_recall.ipynb  # Jupyter notebook for validation analysis
├── Data/
│   └── basic/
│       ├── 1_journal_infos.json       # Journal metadata (tier 1)
│       ├── 2_journal_infos.json       # Journal metadata (tier 2)
│       ├── 3_journal_infos.json       # Journal metadata (tier 3)
│       ├── 4_journal_infos.json       # Journal metadata (tier 4)
│       ├── 4__journal_infos.json      # Additional tier 4 journals
│       └── ABS_Journal_Ranking_2024.xlsx  # ABS journal rankings
└── README.md
```

## Key Components

### 1. AI Paper Recognition (`local_vllm.py`)

A Python script that uses vLLM to classify academic paper abstracts as AI-related or not. Features:

- **Model Support**: Configurable for multiple LLMs (Llama 3.1 8B, Qwen 2.5 32B/72B, Llama 3.3 70B)
- **Quantization Options**: Supports AWQ quantization for efficient inference
- **Multi-GPU**: Tensor parallelism support for large models
- **Batch Processing**: Efficient batch inference for large datasets

**AI Detection Criteria:**
The system identifies papers mentioning:
- General AI terms (machine learning, deep learning, NLP, computer vision)
- Specific algorithms (BERT, LSTM, CNN, transformers)
- Classical ML methods (decision trees, SVM, k-means)
- AI applications (recommendation systems, chatbots)
- Statistical learning methods (Bayesian models, Monte Carlo, Markov models)
- Text/image analysis, sentiment analysis, clustering

### 2. Validation Analysis (`validate_recall.ipynb`)

Jupyter notebook for:
- Comparing LLM predictions against GPT-labeled ground truth
- Analyzing recall and precision metrics
- Identifying AI methods mentioned in papers
- Cross-referencing multiple model outputs

### 3. Journal Data

The `Data/basic/` directory contains:
- **Journal Info JSONs**: OpenAlex API URLs for journal metadata organized by tiers
- **ABS Rankings**: 2024 ABS Journal Rankings (4*, 4, 3, 2, 1 star ratings)

## Usage

### Setting up the environment

```bash
# Set data directory environment variable
export DATA=/path/to/data
```

### Running inference

```python
# In local_vllm.py, select your model configuration:
settings = [
    # Example: Qwen 2.5 72B AWQ (2x A40 GPUs)
    {
        'model_id': f"{system_data_dir}/datasets/llama3_weight/Qwen2.5-72B-Instruct-AWQ",
        'dtype': "float16",
        'quantization': 'awq',
        'tensor_parallel_size': 2,
        'tar_file': "../data/extract_by_llm/Qwen_72B_AWQ_v3/compare_gpt.json",
        'distributed_executor_backend': 'mp',
        'enforce_eager': True,
    },
]

# Run the script
python local_vllm.py
```

### Processing journal rankings

The code processes papers by ABS journal tiers:
- 4* (highest tier)
- 4
- 3
- 2
- 1

## Dependencies

- Python 3.11+
- PyTorch
- vLLM
- pandas
- tqdm
- Jupyter (for validation notebook)

## Research Context

This project is part of research on the evolution and adoption of AI methods in management scholarship. The goal is to track how AI techniques have been incorporated into management research over time, across different journal tiers and research domains.

## Notes

- The project uses OpenAlex API for journal metadata
- Ground truth labels are derived from GPT-based classification
- Multiple model configurations are tested for optimal accuracy/efficiency trade-offs
- Results are stored in JSON format for further analysis

## License

[Add appropriate license information]

## Contact

[Add contact information for the research team]
