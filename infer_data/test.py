from datasets import load_dataset

ds = load_dataset("json",data_files="iter_dpo/Test1_LLaMA3_iter1_data.jsonl",split='train')