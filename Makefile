.PHONY: download_data extract_data

PYTHON = python3


download-data:
	$(PYTHON) src/data/download_data.py


extract-data:
	$(PYTHON) src/data/extract_data.py default-extract


train-tokenizer:
	$(PYTHON) src/features/train_tokenizer.py models/tokenizer_en_de.json \
											  data/processed/train.en \
											  data/processed/train.de


index-train-data:
	$(PYTHON) src/features/index_data.py data/processed/train.en \
									 	 models/tokenizer_en_de.json \
									 	 data/processed/train.en.index

	$(PYTHON) src/features/index_data.py data/processed/train.de \
										 models/tokenizer_en_de.json \
										 data/processed/train.de.index

index-dev-data:
	$(PYTHON) src/features/index_data.py data/processed/dev.en \
									 	 models/tokenizer_en_de.json \
									 	 data/processed/dev.en.index

	$(PYTHON) src/features/index_data.py data/processed/dev.de \
										 models/tokenizer_en_de.json \
										 data/processed/dev.de.index

index-test-data:
	$(PYTHON) src/features/index_data.py data/processed/test_full.en \
									 	 models/tokenizer_en_de.json \
									 	 data/processed/test_full.en.index

	$(PYTHON) src/features/index_data.py data/processed/test_full.de \
									 	 models/tokenizer_en_de.json \
									 	 data/processed/test_full.de.index

	$(PYTHON) src/features/index_data.py data/processed/test_filtered.en \
									 	 models/tokenizer_en_de.json \
									 	 data/processed/test_filtered.en.index

	$(PYTHON) src/features/index_data.py data/processed/test_filtered.de \
									 	 models/tokenizer_en_de.json \
									 	 data/processed/test_filtered.de.index
