import os
import pytorch_lightning as pl


class LRSchedulerVanilla:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.num_step = 0

    def step(self):
        self.num_step += 1
        new_lr = self.d_model ** -0.5 * min(
            self.num_step ** -0.5,
            self.num_step * self.warmup_steps ** -1.5
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class LRSchedulerNew:
    def __init__(self, optimizer, max_learning_rate, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_learning_rate = max_learning_rate
        self.num_step = 0

    def step(self):
        self.num_step += 1

        factor = self.max_learning_rate * self.warmup_steps ** 0.5

        new_lr = factor * min(
            self.num_step ** -0.5,
            self.num_step * self.warmup_steps ** -1.5
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class Checkpointer(pl.Callback):
    def __init__(self, checkpoint_dir: str, checkpoint_every_n_batches: int, save_last_k: int):
        self.checkpoint_every_n_batches = checkpoint_every_n_batches
        self.save_last_k = save_last_k
        self.last_k_checkpoint_paths = []
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_counter = 0
        self.batch_counter = 0

    @pl.utilities.rank_zero_only  # We save checkpoints only on zero-ranked worker
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs) -> None:
        if (self.batch_counter + 1) % self.checkpoint_every_n_batches == 0:
            self.save_checkpoint(trainer)
            if len(self.last_k_checkpoint_paths) > self.save_last_k:
                self.delete_oldest_checkpoint()
        self.batch_counter += 1

    def save_checkpoint(self, trainer):
        file_name = f'{self.checkpoint_counter}.ckpt'
        path = os.path.join(self.checkpoint_dir, file_name)
        trainer.save_checkpoint(path)
        self.last_k_checkpoint_paths.append(path)
        self.checkpoint_counter += 1

    def delete_oldest_checkpoint(self):
        oldest_checkpoint_path = self.last_k_checkpoint_paths.pop(0)
        os.remove(oldest_checkpoint_path)
