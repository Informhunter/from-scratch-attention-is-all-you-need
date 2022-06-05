.PHONY: download-data extract-data train-tokenizer index-data

PYTHON = python3
TOKENIZER_PATH = models/tokenizer_en_de_no_norm.json


download-data:
	$(PYTHON) src/data/download_data.py


extract-data:
	$(PYTHON) src/data/extract_data.py default-extract


train-tokenizer:
	$(PYTHON) src/features/train_tokenizer.py data/processed/train.en \
											  data/processed/train.de \
											  $(TOKENIZER_PATH)


index-data:
	$(PYTHON) src/features/index_data.py data/processed/*.en $(TOKENIZER_PATH)
	$(PYTHON) src/features/index_data.py data/processed/*.de $(TOKENIZER_PATH)
