.PHONY: download_data \
        extract-data \
        train-tokenizer \
        index-data \
        train-base-model \
        test \
        build-training-image

PYTHON := python3

DEVICES := 0,1

PROJECT_NAME := attention-is-all-you-need
TRAINING_IMAGE_NAME := aiayn-training
TRAINING_IMAGE_URI := eu.gcr.io/$(PROJECT_NAME)/$(TRAINING_IMAGE_NAME):latest

SOURCE_LANG := en
TARGET_LANG := de

RAW_DATA = data/raw/training-parallel-nc-v9.tgz \
           data/raw/training-parallel-europarl-v7.tgz \
           data/raw/training-parallel-commoncrawl.tgz \
           data/raw/test-full.tgz \
           data/raw/test-filtered.tgz \
           data/raw/dev.tgz

PROCESSED_DATA = data/processed/train.en \
                 data/processed/train.de \
                 data/processed/dev.en \
                 data/processed/dev.de \
                 data/processed/test_full.en \
                 data/processed/test_full.de \
                 data/processed/test_filtered.en \
                 data/processed/test_filtered.de

DATA_INDEXES = data/processed/train.en.index \
               data/processed/train.de.index \
               data/processed/dev.en.index \
               data/processed/dev.de.index \
               data/processed/test_full.en.index \
               data/processed/test_full.de.index \
               data/processed/test_filtered.en.index \
               data/processed/test_filtered.de.index

TOKENIZER_PATH = models/tokenizer_en_de.json

TRAIN_INDEXES = data/processed/train.$(SOURCE_LANG).index  data/processed/train.$(TARGET_LANG).index
DEV_INDEXES = data/processed/dev.$(SOURCE_LANG).index  data/processed/dev.$(TARGET_LANG).index


# $(RAW_DATA): download-data
$(RAW_DATA): download-data-gcs
$(PROCESSED_DATA): extract-data
$(TOKENIZER_PATH): train-tokenizer
$(DATA_INDEXES): index-data


download-data:
	$(PYTHON) src/data/download_data.py


download-data-gcs:
	gsutil cp gs://project-aiayn/data/raw/* data/raw


extract-data: $(RAW_DATA)
	$(PYTHON) src/data/extract_data.py default-extract


train-tokenizer: data/processed/train.$(SOURCE_LANG) data/processed/train.$(TARGET_LANG)
	$(PYTHON) src/features/train_tokenizer.py data/processed/train.$(SOURCE_LANG) \
	                                          data/processed/train.$(TARGET_LANG) \
	                                          $(TOKENIZER_PATH)


index-data: $(TOKENIZER_PATH) $(PROCESSED_DATA)
	$(PYTHON) src/features/index_data.py $(PROCESSED_DATA) $(TOKENIZER_PATH)


train-base-model: OUTPUT_DIR=models/base_model/
train-base-model: TRAIN_NUM_WORKERS=16
train-base-model: TRAIN_PREFETCH_FACTOR=2
train-base-model: TRAIN_BATCH_SIZE=2500
train-base-model: $(TRAIN_INDEXES) $(DEV_INDEXES)
	$(PYTHON) src/models/train_transformer.py train $(TOKENIZER_PATH) \
	                                                $(TRAIN_INDEXES) \
	                                                $(DEV_INDEXES) \
	                                                $(OUTPUT_DIR) \
	                                                --devices $(DEVICES) \
	                                                --config-path ./model_configs/config_base.json \
	                                                --train-num-workers $(TRAIN_NUM_WORKERS) \
	                                                --train-prefetch-factor $(TRAIN_PREFETCH_FACTOR) \
	                                                --train-batch-size $(TRAIN_BATCH_SIZE)


build-training-image:
	docker build --no-cache -f ./docker/training.dockerfile -t $(TRAINING_IMAGE_NAME) .
	docker image tag $(TRAINING_IMAGE_NAME) $(TRAINING_IMAGE_URI)


devbash-gpu:
	docker run \
		-v $(PWD):"/training" \
		--rm \
		--env PYTHONPATH=/training \
		--env NVIDIA_VISIBLE_DEVICES=all \
		--env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
		--gpus all \
		-ti $(TRAINING_IMAGE_NAME) /bin/bash


test:
	$(PYTHON) -m unittest
