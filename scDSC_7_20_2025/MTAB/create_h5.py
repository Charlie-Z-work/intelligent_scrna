import numpy as np
import h5py

# 读取预处理后的数据
X = np.loadtxt("data/mtab_processed.txt")  # (1193, 5000)
y = np.loadtxt("data/mtab_processed_label.txt", dtype=int)  # (1193,)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# 创建h5文件
with h5py.File("data/mtab.h5", "w") as f:
    f.create_dataset("X", data=X)
    f.create_dataset("Y", data=y)

print("✅ 已创建 mtab.h5 文件")