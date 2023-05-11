from PIL import Image
from skimage import color, morphology, measure
import numpy as np
import os
import matplotlib.pyplot as plt
PATH = '/home/yakov/Downloads/DIC-C2DH-HeLa/01_GT/SEG/'


def process_file(filename):
    gt_file = Image.open(PATH+filename)
    gt_file_np_arr = np.array(gt_file)
    min_label = np.min(gt_file_np_arr)
    max_label = np.max(gt_file_np_arr)
    #labels_image_before = (255 * color.label2rgb(gt_file_np_arr)).astype(np.uint8)
    #labels_image_after = np.copy(labels_image_before)
    cell_edges_map = np.zeros_like(gt_file_np_arr, dtype=np.uint8)
    distances_map = np.zeros_like((gt_file_np_arr.shape[0], gt_file_np_arr.shape[1], max_label+1), np.float32)
    for k in range(min_label+1, max_label+1):
        kth_blob = gt_file_np_arr == k
        eroded_blob = morphology.erosion(kth_blob)
        diff_image = np.logical_and(kth_blob, np.logical_not(eroded_blob))
        twice_eroded_blob = morphology.erosion(eroded_blob)
        kth_contour = np.logical_and(eroded_blob, np.logical_not(twice_eroded_blob))
        kth_contour
        gt_file_np_arr[diff_image] = 0
        cell_edges_map[kth_contour] = k
        plt.imshow(kth_contour)



if __name__ == "__main__":
    files = os.listdir(PATH)
    files = list(filter(lambda x: 'tif' in x, files))
    print(files)
    for file in files[:1]:
        process_file(file)

        #np.save(PATH+file.split(".")[0], gt_file_np_arr)

            #i1 = Image.fromarray(labels_image)
            #i1 = Image.fromarray(diff_image)
            #i2 = Image.fromarray(eroded_label_image)
            #i1.show()
            #i2.show()



    print("done")