.PHONY: extract-data \
        train-tokenizer \
        index-data \
        train-model \
        test \
        build-training-image

PYTHON := python3

DEVICES = 0,1

PROJECT_NAME := attention-is-all-you-need
TRAINING_IMAGE_NAME := aiayn-training
TRAINING_IMAGE_URI := eu.gcr.io/$(PROJECT_NAME)/$(TRAINING_IMAGE_NAME):latest
GCP_PREFIX := gs://project-aiayn

SOURCE_LANG := en
TARGET_LANG := de


# Possible values gcs_processed/gcs_raw/original
FETCH_DATA_MODE := original


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

TRAIN_SOURCE_INDEX = data/processed/train.$(SOURCE_LANG).index
TRAIN_TARGET_INDEX = data/processed/train.$(TARGET_LANG).index
DEV_SOURCE_INDEX = data/processed/dev.$(SOURCE_LANG).index
DEV_TARGET_INDEX = data/processed/dev.$(TARGET_LANG).index
TEST_SOURCE = data/processed/test_filtered.$(SOURCE_LANG)
TEST_TARGET = data/processed/test_filtered.$(TARGET_LANG)



# Commands

define download-raw-data
	$(PYTHON) src/data/download_data.py
endef

define download-raw-data-gcs
	gsutil cp $(GCP_PREFIX)/data/raw/* data/raw
endef

define download-processed-data-gcs
	gsutil cp $(GCP_PREFIX)/data/processed/* data/processed
	gsutil cp $(GCP_PREFIX)/models/tokenizer_en_de.json $(TOKENIZER_PATH)
endef

define extract-data
	$(PYTHON) -m src.data.extract_data default-extract
endef

define train-tokenizer
	$(PYTHON) -m src.features.train_tokenizer data/processed/train.$(SOURCE_LANG) \
	                                          data/processed/train.$(TARGET_LANG) \
	                                          $(TOKENIZER_PATH)
endef

define index-data
	$(PYTHON) -m src.features.index_data $(PROCESSED_DATA) $(TOKENIZER_PATH)
endef

# Download processed data (including tokenizer and data indexes) from GCS
ifeq ($(FETCH_DATA_MODE), gcs_processed)
$(PROCESSED_DATA) $(TOKENIZER_PATH) $(DATA_INDEXES):
	$(call download-processed-data-gcs)

# Download only raw data from GCS
else ifeq ($(FETCH_DATA_MODE), gcs_raw)
$(RAW_DATA):
	$(call download-raw-data-gcs)
$(PROCESSED_DATA): $(RAW_DATA)
	$(call extract-data)
$(TOKENIZER_PATH): $(PROCESSED_DATA)
	$(call train-tokenizer)
$(DATA_INDEXES): $(PROCESSED_DATA) $(TOKENIZER_PATH)
	$(call index-data)

# Download original raw data from WMT website
else ifeq ($(FETCH_DATA_MODE), original)
$(RAW_DATA):
	$(call download-raw-data)
$(PROCESSED_DATA): $(RAW_DATA)
	$(call extract-data)
$(TOKENIZER_PATH): $(PROCESSED_DATA)
	$(call train-tokenizer)
$(DATA_INDEXES): $(PROCESSED_DATA) $(TOKENIZER_PATH)
	$(call index-data)
endif

train-model: OUTPUT_DIR=models/base_model/
train-model: TRAIN_CONFIG_PATH=model_configs/config_base.json
train-model: $(TRAIN_SOURCE_INDEX) $(TRAIN_TARGET_INDEX) $(DEV_SOURCE_INDEX) $(DEV_TARGET_INDEX) $(TOKENIZER_PATH)
	$(PYTHON) -m src.models.transformer train --tokenizer $(TOKENIZER_PATH) \
	                                          --train-source-index $(TRAIN_SOURCE_INDEX) \
	                                          --train-target-index $(TRAIN_TARGET_INDEX) \
	                                          --val-source-index $(DEV_SOURCE_INDEX) \
	                                          --val-target-index $(DEV_TARGET_INDEX) \
	                                          --output-dir $(OUTPUT_DIR) \
	                                          --devices $(DEVICES) \
	                                          --config $(TRAIN_CONFIG_PATH) \
	                                          $(TRAIN_ARGS)


test-model: CHECKPOINT_PATH=models/model.ckpt
test-model:
	$(PYTHON) -m src.models.transformer test --regular-checkpoint $(CHECKPOINT_PATH) \
	                                         --source $(TEST_SOURCE) \
	                                         --target $(TEST_TARGET) \
	                                         --devices $(DEVICES)


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
