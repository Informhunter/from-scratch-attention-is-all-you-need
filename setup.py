from setuptools import setup, find_packages

setup(
    name='src',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch==1.9.0+cu111',
        'pytorch-lightning',
        'jupyterlab',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'tokenizers',
        'sacrebleu',
        'click',
        'tqdm',
        'aiohttp',
        'optuna',
        'fsspec[gcs]',
        'wandb',
    ]
)
