from setuptools import setup, find_packages

setup(
    name='src',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch==1.11.0+cu113',
        'pytorch-lightning==1.6.4',
        'jupyterlab',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn==1.1.1',
        'tokenizers==0.12.1',
        'sacrebleu==2.1.0',
        'click',
        'tqdm',
        'aiohttp',
        'optuna',
        'fsspec[gcs]',
        'wandb',
    ]
)
