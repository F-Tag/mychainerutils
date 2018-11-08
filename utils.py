
import numpy as np


def get_shuffled_example(length):
    origin = np.arange(length)
    shuffled = np.random.permutation(origin)
    dup_index = np.where(origin==shuffled)[0].tolist()
    dup = len((dup_index))
    if dup == 1:
        while True:
            idx = np.random.randint(length)
            if idx != dup_index[0]:
                break
        shuffled[dup_index + [idx]] = shuffled[[idx] + dup_index]
    elif dup == 2:
        shuffled[dup_index] = shuffled[dup_index][::-1]
    elif dup >= 3:
        ch_s_index = get_shuffled_example(dup)
        shuffled[dup_index] = shuffled[dup_index][ch_s_index]
    
    return shuffled