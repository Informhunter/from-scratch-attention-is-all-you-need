import asyncio
from typing import (
    List,
    Tuple,
)

import aiohttp
from tqdm.asyncio import tqdm


async def download_file(url: str, output_path: str, session: aiohttp.ClientSession):
    async with session.get(url, timeout=None) as response:
        with open(output_path, 'wb') as f:
            with tqdm(response.content.iter_chunked(1024), unit='KB') as pbar:
                pbar.set_description(f'Downloading: "{url}" into "{output_path}"')
                async for data in pbar:
                    f.write(data)
                pbar.close()


async def download_files(urls_paths: List[Tuple[str, str]]):
    async with aiohttp.ClientSession() as session:
        tasks = [download_file(url, path, session) for url, path in urls_paths]
        return await asyncio.gather(*tasks, return_exceptions=True)


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(download_files(
        [
            ('http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
             'data/raw/training-parallel-europarl-v7.tgz'),
            ('http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
             'data/raw/training-parallel-commoncrawl.tgz'),
            ('http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz',
             'data/raw/training-parallel-nc-v9.tgz'),
            ('http://www.statmt.org/wmt14/dev.tgz',
             'data/raw/dev.tgz'),
            ('http://www.statmt.org/wmt14/test-filtered.tgz',
             'data/raw/test-filtered.tgz'),
            ('http://statmt.org/wmt14/test-full.tgz',
             'data/raw/test-full.tgz'),
        ]
    ))


if __name__ == '__main__':
    main()
