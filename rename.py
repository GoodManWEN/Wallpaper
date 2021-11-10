import os
import re
import base64
import hashlib
import numpy as np
from PIL import Image
from pipeit import *


WATCH_LIST = ['.png','.jpg','.jpeg','.bmp','.webp']
WSIZE, HSIZE = 16, 9
MIN_HASH_LENGTH = (WSIZE * HSIZE) * 6 // 8
MIN_HASH_LENGTH_BIN = (WSIZE * HSIZE) * 6
CHAR_PATT = re.compile('[a-zA-Z0-9+\-]+')

def img_phash_bin_parser(file_path: str) -> str:
    _ = os.path.split(file_path)[1]
    _, ext1 = os.path.splitext(_)
    if ext1 != '':
        name, ext2 = os.path.splitext(_)
        if ext2 == '.dc' and len(name) > MIN_HASH_LENGTH and CHAR_PATT.match(name).end() == len(name):
            # 合法的phash格式
            return bin(int.from_bytes(base64.b64decode(name.replace('-','/')), byteorder='big'))[2:].zfill(MIN_HASH_LENGTH_BIN)
    return False

def img_loader_and_rgb_resize(file_path: str) -> np.ndarray:
    img = Image.open(file_path)
    swidth, sheight = img.size
    if swidth >= sheight :
        nwidth, nheight = WSIZE, HSIZE
    else:
        nwidth, nheight = HSIZE, WSIZE
    img_rgb = img.convert("RGB")
    img_resized = img_rgb.resize((nwidth, nheight), Image.BICUBIC)
    img_matrix = np.asarray(img_resized)
    return img_matrix

def compute_perceptual_hash(array: np.ndarray) -> str:
    array = array.astype(np.int16)
    h_diff = array - np.roll(array, 1, axis = 1)
    h_diff[h_diff >= 0] = 1
    h_diff = np.maximum(h_diff, 0)
    v_diff = array - np.roll(array, 1, axis = 0)
    v_diff[v_diff >= 0] = 1
    v_diff = np.maximum(v_diff, 0)
    phash = np.concatenate((
        np.vstack((h_diff[:,:,0], h_diff[:,:,1], h_diff[:,:,2])).reshape(-1),
        np.vstack((v_diff[:,:,0], v_diff[:,:,1], v_diff[:,:,2])).reshape(-1)
    ), axis = 0)
    return base64.b64encode(bytearray(np.packbits(phash))).decode()

def hamming_distance(str1, str2) -> int:
    if len(str1) != len(str2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(str1, str2))

def imge_value_compute(file_path1: str, file_path2: str) -> int:
    # 保留体积大的，分辨率高的，webp优先
    img1 = Image.open(file_path1)
    img2 = Image.open(file_path2)
    img1_w_v = max(img1.size[0] / img2.size[0], 1)
    img2_w_v = max(img2.size[0] / img1.size[0], 1)
    img1_h_v = max(img1.size[1] / img2.size[1], 1)
    img2_h_v = max(img2.size[1] / img1.size[1], 1)
    img1_res_v = img1_w_v * img1_h_v
    img2_res_v = img2_w_v * img2_h_v

    img1_fs = os.stat(file_path1).st_size
    img2_fs = os.stat(file_path2).st_size
    img1_sz_v = max(((img1_fs - img2_fs) / min(img1_fs, img2_fs)) * 0.8, 0) + 1
    img2_sz_v = max(((img2_fs - img1_fs) / min(img1_fs, img2_fs)) * 0.8, 0) + 1
    img1_webp_v = 100 if os.path.splitext(file_path1)[1] == '.webp' else 0
    img2_webp_v = 100 if os.path.splitext(file_path2)[1] == '.webp' else 0
    img1_v = img1_res_v * img1_sz_v + img1_webp_v
    img2_v = img2_res_v * img2_sz_v + img2_webp_v
    if img1_v >= img2_v:
        return 1 
    else:
        return 0

class PhashSet(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add(self, key, value) -> bool:
        sim_key = self._exists_file(key)
        if sim_key:
            orn_img_path = self[sim_key]
            new_img_path = value 
            which_one_to_delete = imge_value_compute(orn_img_path, new_img_path)
            if which_one_to_delete == 1:
                # 删除新的
                return False # 未添加成功
            else:
                # 删除旧的
                os.remove(orn_img_path)
                del self[sim_key]
                self[key] = value
                return True
        else:
            self[key] = value
            return True

    def _exists_file(self, phash: str) -> str:
        for key in tuple(self.keys()):
            if hamming_distance(key, phash) <= 32:
                return key
        return False

dir_list = ['.', ]
phash_set = PhashSet()
print('Duplication checking...', end='')
while dir_list:
    _main_dir = dir_list.pop()
    for _path in os.walk(_main_dir):
        _sub_dir = _path[1]
        _files = _path[2]
        break
    dir_list.extend(_sub_dir | 
        Filter(lambda x:len(x) > 0 and x[0] != '.') |
        Map(lambda x:os.path.join(_main_dir , x)) | 
        list
    )
    _files = (_files | 
        Filter(lambda x:len(x) > 0 and x[0] != '.') |
        Filter(lambda x:os.path.splitext(x)[1].lower() in WATCH_LIST) | 
        # Filter(lambda x:len(os.path.split(os.path.splitext(x)[0])[1]) >= 108) | 
        # Map(lambda x:os.path.join(_main_dir , x)) | 
        list
    )
    _files.sort(key = lambda x:os.stat(os.path.join(_main_dir , x)).st_size, reverse = True)
    for _file in _files:
        name , ext = os.path.splitext(_file)
        file_path_full = os.path.join(_main_dir, _file)
        phash_bin = img_phash_bin_parser(file_path_full)
        if phash_bin:
            # 已扫描过的文件
            if not phash_set.add(phash_bin, file_path_full):
                os.remove(file_path_full)
        else:
            # 未经扫描的文件
            img_phash = compute_perceptual_hash(img_loader_and_rgb_resize(file_path_full))
            img_phash_bin = bin(int.from_bytes(base64.b64decode(img_phash), byteorder='big'))[2:].zfill(MIN_HASH_LENGTH_BIN)
            new_name = f"{img_phash.replace('/', '-')}.dc{ext}"
            new_name_full = os.path.join(_main_dir, new_name)
            if not phash_set.add(img_phash_bin, file_path_full):
                # 添加失败，重复文件
                os.remove(file_path_full)
            else:
                try:
                    os.rename(file_path_full, new_name_full)
                except Exception as e:
                    if isinstance(e , FileExistsError):
                        # 重命名失败：已存在同名文件
                        os.remove(_file)
        print('.', end='')
print('\nDone.')