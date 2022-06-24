# From Scratch Attention Is All You Need

---

Implementation of the Attention Is All You Need paper https://arxiv.org/abs/1706.03762

Mainly experimented with training base model for the task of English-German translation.


## Usage

### Docker

Build image
```shell
make build-training-image
```
Launch dev shell inside docker container
```shell
make devbash-gpu
```

### Inside docker container

Train base model
```shell
make train-model
```

Train big model
```shell
make train-model TRAIN_CONFIG_PATH=model_configs/big_config.json
```

Train base model and gradient accumulation over 4 batches, use GPUs 0 and 1
```shell
make train-model TRAIN_CONFIG_PATH=model_configs/base_config.json \
                 TRAIN_ARGS="--config__trainer__accumulate_grad_batches=4" \
                 DEVICES="0,1"
```

Test a model on the default test set
```shell
make test-model CHECKPOINT_PATH=""
```

Test a model on a custom test set
```shell
make test-model CHECKPOINT_PATH="" \
                TEST_SOURCE="" \
                TEST_TARGET=""
```


## References

- Joey NMT (fast beam search implementation) https://github.com/joeynmt/joeynmt
- Attention Is All You Need https://arxiv.org/abs/1706.03762
- Training Tips for the Transformer Model https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf
