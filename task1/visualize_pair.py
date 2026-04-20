"""
可视化：原始X光片 + Ground Truth Mask + 模型预测 Mask
生成对比图保存到 log/dicece2d/visualization/
"""
import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 路径配置
LOG_DIR = 'log/dicece2d'
DATA_CSV = 'dataprocess/data/valid2d.csv'
OUTPUT_DIR = os.path.join(LOG_DIR, 'visualization')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取一张验证集图片和对应mask
import pandas as pd
csv_data = pd.read_csv(DATA_CSV, header=None)
img_path = csv_data.iloc[0, 0]
mask_path = csv_data.iloc[0, 1]

# 读取原图和mask
src_image = cv2.imread(img_path, 0)
gt_mask = cv2.imread(mask_path, 0)

# 读取模型训练过程中保存的预测结果（取最后一个epoch和第1个epoch对比）
epochs_to_show = [1, 50, 100, 250, 500]
epochs_to_show = [e for e in epochs_to_show if os.path.exists(os.path.join(LOG_DIR, f'{e}_Val_EPOCH_pdmask.png'))]

# === 图1: 原始数据 pair (X光片 + GT mask) ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(src_image, cmap='gray')
axes[0].set_title('Original X-ray', fontsize=14)
axes[0].axis('off')

axes[1].imshow(gt_mask, cmap='nipy_spectral', vmin=0, vmax=52)
axes[1].set_title('Ground Truth (53 classes)', fontsize=14)
axes[1].axis('off')

# overlay
overlay = cv2.cvtColor(src_image, cv2.COLOR_GRAY2RGB)
mask_color = plt.cm.nipy_spectral(gt_mask.astype(float) / 52)[:, :, :3]
mask_color = (mask_color * 255).astype(np.uint8)
overlay_resized = cv2.resize(overlay, (gt_mask.shape[1], gt_mask.shape[0]))
blended = cv2.addWeighted(overlay_resized, 0.6, mask_color, 0.4, 0)
axes[2].imshow(blended)
axes[2].set_title('Overlay (X-ray + GT)', fontsize=14)
axes[2].axis('off')

plt.suptitle('Training Data Pair: X-ray + Tooth Segmentation Label', fontsize=16, fontweight='bold')
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'data_pair.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path}')

# === 图2: 训练进度（GT vs Prediction at different epochs） ===
if epochs_to_show:
    n = len(epochs_to_show)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, ep in enumerate(epochs_to_show):
        gt_path = os.path.join(LOG_DIR, f'{ep}_Val_EPOCH_gtmask.png')
        pd_path = os.path.join(LOG_DIR, f'{ep}_Val_EPOCH_pdmask.png')
        gt_img = cv2.imread(gt_path, 0) if os.path.exists(gt_path) else np.zeros((512, 512), dtype=np.uint8)
        pd_img = cv2.imread(pd_path, 0) if os.path.exists(pd_path) else np.zeros((512, 512), dtype=np.uint8)

        axes[0, i].imshow(gt_img, cmap='nipy_spectral')
        axes[0, i].set_title(f'GT (epoch {ep})', fontsize=12)
        axes[0, i].axis('off')

        axes[1, i].imshow(pd_img, cmap='nipy_spectral')
        axes[1, i].set_title(f'Prediction (epoch {ep})', fontsize=12)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Ground Truth', fontsize=13, fontweight='bold')
    axes[1, 0].set_ylabel('Prediction', fontsize=13, fontweight='bold')
    plt.suptitle('Training Progress: GT vs Model Prediction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path2 = os.path.join(OUTPUT_DIR, 'training_progress.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path2}')

print(f'\nAll visualizations saved to: {os.path.abspath(OUTPUT_DIR)}')
