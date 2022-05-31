import re
from pathlib import Path

from datasets import load_dataset


def prepare_data(overwrite_cache=False, debug=False):
    folder = "data"
    if Path(folder).exists() and not overwrite_cache:
        print("data folder already exists. Set the flag overwrite_cache to True to download the data again.")
        return

    Path(folder).mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("cuad")

    for split in ['train', 'test']:
        text = dataset[split]['context']
        if debug:
            text = text[:100]

        with open(f"data/{split}.txt", 'w+') as output:
            for row in text:
                row = re.sub(r'\n+', '\n', row).strip()
                output.write(str(row) + '\n')
    print(f"Saved the data to the folder {folder}")


if __name__ == '__main__':
    prepare_data()
