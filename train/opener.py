# coding=utf-8

"""Helper for opening multiple results files at the same time."""

import os

source_dir = "G:\\"
all_dirs = os.listdir(source_dir)
target_dirs = [os.path.join(source_dir, d) for d in all_dirs if "uvnet-net-j" in d]

for d in sorted(target_dirs):
    os.startfile(os.path.join(d, "results.html"))
