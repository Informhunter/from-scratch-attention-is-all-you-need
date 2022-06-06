from typing import List


def parse_devices(_, __, devices: str) -> List[int]:
    return [int(x.strip()) for x in devices.split(',')]
