import os
import cv2
from tqdm import trange

print('\033[0;0f\033[0J')
root_dir = '/media/SSD1/cityscapes/'

data_dir = os.path.join(root_dir, 'leftImg8bit/train/')
label_dir = os.path.join(root_dir, 'gtFine/train/')

def transform(img):
    center_y = 512
    center_x = 1024
    y1, y2 = center_y - 3*center_y//4, center_y + 3*center_y//4
    x1, x2 = center_x - 3*center_x//4, center_x + 3*center_x//4
    r_img = img[y1:y2, x1:x2]

    f_img = cv2.flip(img, 0)        # Horizontal flip
    rf_img = cv2.flip(r_img, 0)     # Horizontal flip cropped img

    return r_img, f_img, rf_img


pbar1 = trange(len(os.listdir(data_dir)), position=0, desc='Overall progress ')
for folder in os.listdir(data_dir):
    d = os.path.join(data_dir, folder)
    if not os.path.isdir(d):
        continue

    pbar2 = trange(len(os.listdir(d)), position=1, desc='Within folder progress ')
    for filename in os.listdir(d):
        if filename.endswith('.png'):
            data_path = '{0}/{1}/{2}'.format(data_dir, folder, filename)
            label_file = filename.replace('leftImg8bit', 'gtFine_labelIds')
            label_path = '{0}/{1}/{2}'.format(label_dir, folder, label_file)

            source_img = cv2.imread(data_path)
            r_img, f_img, rf_img = transform(source_img)
            dest_path = data_path[:-4] + '_r.png'
            cv2.imwrite(dest_path, r_img)
            dest_path = data_path[:-4] + '_f.png'
            cv2.imwrite(dest_path, f_img)
            dest_path = data_path[:-4] + '_rf.png'
            cv2.imwrite(dest_path, rf_img)

            source_img = cv2.imread(label_path, 0)
            r_img, f_img, rf_img = transform(source_img)
            dest_path = label_path[:-4] + '_r.png'
            cv2.imwrite(dest_path, r_img)
            dest_path = label_path[:-4] + '_f.png'
            cv2.imwrite(dest_path, f_img)
            dest_path = label_path[:-4] + '_rf.png'
            cv2.imwrite(dest_path, rf_img)

        pbar2.update(1)

    pbar2.close()
    pbar1.update(1)

pbar1.close()
print('\nData augmentation complete')
