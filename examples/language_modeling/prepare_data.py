import re
from pathlib import Path

from datasets import load_dataset

dataset = load_dataset("cuad")

Path("data").mkdir(parents=True, exist_ok=True)

for split in ['train', 'test']:
    text = dataset[split]['context']

    with open(f"data/{split}.txt", 'w+') as output:
        for row in text[:10]:
            row = re.sub(r'\n+', '\n', row).strip()
            output.write(str(row) + '\n')
