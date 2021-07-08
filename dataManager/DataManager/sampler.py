"""
负采样工具
"""

from random import choices

def unine_sampler(all_items,history,ratio):
    remain=list(
        set(all_items)-set(history)
    )
    l=len(history)*ratio
    return choices(remain,k=l)
