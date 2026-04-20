"""
数据预处理脚本：
1. 把 Train-Labeled 的 JSON 标注转成 mask 图片
2. 划分 train/valid 集 (25 train + 5 valid)
3. 生成 CSV 索引文件供训练脚本使用
"""
import os
import cv2
import json
import numpy as np

# ============ 路径配置 ============
# 原始数据路径
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'STS24-2DXray'))
LABELED_DIR = os.path.join(DATA_ROOT, 'Train-Labeled')
IMAGE_DIR = os.path.join(LABELED_DIR, 'Images')
MASK_DIR = os.path.join(LABELED_DIR, 'Masks')

# 预处理输出路径
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), 'dataprocess', 'processed')
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_ROOT, 'Image')
OUTPUT_MASK_DIR = os.path.join(OUTPUT_ROOT, 'Mask')

# CSV 输出路径
CSV_DIR = os.path.join(os.path.dirname(__file__), 'dataprocess', 'data')

# ============ JSON -> Mask 转换（来自 STS2ddataprocess.py） ============
def convertjsontomask(json_data, image):
    label_to_value = {
        11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
        21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
        31: 17, 32: 18, 33: 19, 34: 20, 35: 21, 36: 22, 37: 23, 38: 24,
        41: 25, 42: 26, 43: 27, 44: 28, 45: 29, 46: 30, 47: 31, 48: 32,
        51: 33, 52: 34, 53: 35, 54: 36, 55: 37,
        61: 38, 62: 39, 63: 40, 64: 41, 65: 42,
        71: 43, 72: 44, 73: 45, 74: 46, 75: 47,
        81: 48, 82: 49, 83: 50, 84: 51, 85: 52,
    }
    mask = np.zeros_like(image)
    for shape in json_data['shapes']:
        label = int(shape['label'])
        contours = shape['points']
        if label in label_to_value:
            opencvcontour = np.array(contours).reshape((len(contours), 1, 2)).astype(np.int32)
            cv2.drawContours(mask, [opencvcontour], -1, label_to_value[label], -1)
    return mask


def main():
    # 1. 创建输出目录
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    # 2. 检查数据
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.bmp'))])
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.json')])
    print(f"Found {len(image_files)} images, {len(mask_files)} JSON masks")

    if len(image_files) == 0:
        print(f"ERROR: No images found in {IMAGE_DIR}")
        return

    # 3. 转换 JSON -> Mask 图片
    processed = []
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, img_file)
        # 找对应的 JSON mask
        base_name = os.path.splitext(img_file)[0]
        json_file = base_name + '_Mask.json'
        json_path = os.path.join(MASK_DIR, json_file)

        if not os.path.exists(json_path):
            print(f"  SKIP: No mask for {img_file}")
            continue

        # 读取图片（灰度）
        src_image = cv2.imread(img_path, 0)
        if src_image is None:
            print(f"  SKIP: Cannot read {img_file}")
            continue

        # 读取 JSON 并转换
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        mask = convertjsontomask(json_data, src_image)

        # 保存为 BMP
        out_name = base_name + '.bmp'
        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, out_name), src_image)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, out_name), mask)
        processed.append(out_name)
        print(f"  [{i+1}/{len(image_files)}] {img_file} -> {out_name}  (labels: {np.unique(mask).tolist()})")

    print(f"\nProcessed {len(processed)} image-mask pairs")

    # 4. 划分 train/valid (25/5)
    np.random.seed(42)
    indices = np.random.permutation(len(processed))
    n_valid = 5
    valid_indices = indices[:n_valid]
    train_indices = indices[n_valid:]

    # 5. 生成 CSV
    train_csv = os.path.join(CSV_DIR, 'train2d.csv')
    valid_csv = os.path.join(CSV_DIR, 'valid2d.csv')

    with open(train_csv, 'w') as f:
        for idx in train_indices:
            img_path = os.path.join(OUTPUT_IMAGE_DIR, processed[idx])
            mask_path = os.path.join(OUTPUT_MASK_DIR, processed[idx])
            # 用绝对路径
            f.write(f"{os.path.abspath(img_path)},{os.path.abspath(mask_path)}\n")

    with open(valid_csv, 'w') as f:
        for idx in valid_indices:
            img_path = os.path.join(OUTPUT_IMAGE_DIR, processed[idx])
            mask_path = os.path.join(OUTPUT_MASK_DIR, processed[idx])
            f.write(f"{os.path.abspath(img_path)},{os.path.abspath(mask_path)}\n")

    # 生成空的 unlabel CSV（训练脚本需要）
    unlabel_csv = os.path.join(CSV_DIR, 'unlabel_train2d.csv')
    with open(unlabel_csv, 'w') as f:
        pass  # 暂时留空，后续可以加无标注数据

    print(f"\nCSV files generated:")
    print(f"  Train: {train_csv} ({len(train_indices)} samples)")
    print(f"  Valid: {valid_csv} ({len(valid_indices)} samples)")
    print(f"  Unlabel: {unlabel_csv} (empty)")
    print("\nDone! You can now run trainmutil.py")


if __name__ == '__main__':
    main()
