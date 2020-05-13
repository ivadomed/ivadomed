from ivadomed.utils import segment_volume
import psutil, os

pid = os.getpid()
ps = psutil.Process(pid)
print(ps.memory_info())
segment_volume("/home/andreanne/Documents/models/seg_tumor", "/home/andreanne/Documents/dataset/pad_512x256x16/data/sub-Astr144/anat/sub-Astr144_T2w.nii.gz")
print(ps.memory_info())