.PHONY: download-data extract-data train-tokenizer index-data clean test

SHELL := /bin/bash
.SHELLFLAGS := -O extglob -c

PYTHON := python3
TOKENIZER_PATH := models/tokenizer_en_de_no_norm.json

DEVICES := 0,1


download-data:
	$(PYTHON) src/data/download_data.py


extract-data:
	$(PYTHON) src/data/extract_data.py default-extract


train-tokenizer:
	$(PYTHON) src/features/train_tokenizer.py data/processed/train.en \
	                                          data/processed/train.de \
	                                          $(TOKENIZER_PATH)


index-data:
	$(PYTHON) src/features/index_data.py data/processed/*.!("index") $(TOKENIZER_PATH)


clean:
	rm data/processed/!(".gitignore")


test:
	$(PYTHON) -m unittest


train-base-model: OUTPUT_DIR=./models/base_model/
train-base-model:
	$(PYTHON) src/models/train_transformer.py train ./models/tokenizer_en_de_no_norm.json \
	                                                ./data/processed/train.en.index \
	                                                ./data/processed/train.de.index \
	                                                ./data/processed/dev.en.index \
	                                                ./data/processed/dev.de.index \
	                                                $(OUTPUT_DIR) \
	                                                --devices $(DEVICES) \
	                                                --config-path ./model_configs/config_base.json
