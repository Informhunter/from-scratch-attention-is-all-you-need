import random
import tarfile


def read_lines(tar_file, internal_file):
    with tar_file.extractfile(internal_file) as f:
        data = f.read()
    return data.decode('utf-8').split('\n')


def main():
    train_texts_en = []
    train_texts_de = []

    dev_texts_en = []
    dev_texts_de = []

    with tarfile.open('./data/raw/training-parallel-commoncrawl.tgz', 'r:gz') as t:
        train_texts_en.extend(read_lines(t, 'commoncrawl.de-en.en'))
        train_texts_de.extend(read_lines(t, 'commoncrawl.de-en.de'))

    with tarfile.open('./data/raw/training-parallel-europarl-v7.tgz', 'r:gz') as t:
        train_texts_en.extend(read_lines(t, 'training/europarl-v7.de-en.en'))
        train_texts_de.extend(read_lines(t, 'training/europarl-v7.de-en.de'))

    with tarfile.open('./data/raw/training-parallel-nc-v9.tgz', 'r:gz') as t:
        train_texts_en.extend(read_lines(t, 'training/news-commentary-v9.de-en.en'))
        train_texts_de.extend(read_lines(t, 'training/news-commentary-v9.de-en.de'))

    with tarfile.open('./data/raw/dev.tgz', 'r:gz') as t:
        dev_texts_en.extend(read_lines(t, 'dev/newstest2013.en'))
        dev_texts_de.extend(read_lines(t, 'dev/newstest2013.de'))

    train_pairs = list(zip(train_texts_en, train_texts_de))
    dev_pairs = list(zip(dev_texts_en, dev_texts_de))

    random.seed(123)
    random.shuffle(train_pairs)

    with open('./data/processed/train_en.tsv', 'w', encoding='utf-8') as en_f,\
         open('./data/processed/train_de.tsv', 'w', encoding='utf-8') as de_f:

        for en_text, de_text in train_pairs:
            en_f.write(en_text.strip() + '\n')
            de_f.write(de_text.strip() + '\n')

    with open('./data/processed/dev_en.tsv', 'w', encoding='utf-8') as en_f, \
            open('./data/processed/dev_de.tsv', 'w', encoding='utf-8') as de_f:

        for en_text, de_text in dev_pairs:
            en_f.write(en_text.strip() + '\n')
            de_f.write(de_text.strip() + '\n')


if __name__ == '__main__':
    main()
