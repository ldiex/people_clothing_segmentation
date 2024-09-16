import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

def img_augmentation(img_path, mask_path, simplify_class: bool):
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if simplify_class:
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] in tops_indices:
                    mask[i, j] = 1
                elif mask[i, j] in bottoms_indices:
                    mask[i, j] = 2
                elif mask[i, j] in shoes_indices:
                    mask[i, j] = 3
                elif mask[i, j] in skin_indices:
                    mask[i, j] = 4
                elif mask[i, j] in back_indices:
                    mask[i, j] = 0
        
    # Random Horizontal Flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    # Random Vertical Flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 0)
        mask = cv2.flip(mask, 0)

    # Random Rotation
    if np.random.rand() > 0.5:
        angle = np.random.randint(0, 360)
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

    # Random Lightness
    if np.random.rand() > 0.5:
        alpha = 1.5 + (np.random.rand() - 0.5)
        beta = 0
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Random Contrast
    if np.random.rand() > 0.5:
        alpha = 1
        beta = 50 * (np.random.rand() - 0.5)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Random Perspective
    if np.random.rand() > 0.5:
        pts1 = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
        pts2 = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        mask = cv2.warpPerspective(mask, M, (mask.shape[1], mask.shape[0]))

    return img, mask

# Setting Paths for Images and Masks
img_path = Path('data/png_images/IMAGES')
img_path_list = list(img_path.glob('*.png'))
img_path_list = sorted(img_path_list)

print(f"total images: {len(img_path_list)}")

mask_path = Path('data/png_masks/MASKS')
mask_path_list = list(mask_path.glob('*.png'))
mask_path_list = sorted(mask_path_list)

print(f"total masks: {len(mask_path_list)}")

img_aug_path = Path('data/aug_images')
mask_aug_path = Path('data/aug_masks')

tops_indices = [4, 5, 6, 8, 10, 11, 13, 14, 22, 24, 26, 35, 38, 46, 48, 49, 50, 51, 52, 54, 55]
bottoms_indices = [25, 27, 30, 31, 40, 42, 44, 45, 53]
shoes_indices = [7, 12, 16, 21, 28, 32, 36, 39, 43, 58]
skin_indices = [41]
back_indices = list(filter(lambda x: x not in tops_indices + bottoms_indices + shoes_indices + skin_indices, range(59)))

# Augmenting Images and Masks
for i in tqdm(range(len(img_path_list))):
    img = cv2.imread(str(img_path_list[i]))
    mask = cv2.imread(str(mask_path_list[i]), cv2.IMREAD_GRAYSCALE)
    for j in range(3):
        img_aug, mask_aug = img_augmentation(img_path_list[i], mask_path_list[i], simplify_class=True)

        cv2.imwrite(str(img_aug_path / f'{i}_{j}.png'), img_aug)
        cv2.imwrite(str(mask_aug_path / f'{i}_{j}.png'), mask_aug)


