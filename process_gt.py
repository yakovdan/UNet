from PIL import Image
from skimage import color, morphology, measure
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
PATH = '/home/yakov/Downloads/DIC-C2DH-HeLa/01_GT/SEG/'


def contour_to_min_dist_array(contour):
    contour_pixels = np.vstack(np.where(contour)).T.astype(np.int32)
    image_pixels = np.array([[x, y] for y in range(512) for x in range(512)]).astype(np.int32)
    dist_array = cdist(image_pixels, contour_pixels)
    return image_pixels, np.min(dist_array, axis=1)


def process_file(filename):
    gt_file = Image.open(PATH+filename)
    gt_file_np_arr = np.array(gt_file)
    min_label = np.min(gt_file_np_arr)
    max_label = np.max(gt_file_np_arr)
    cell_edges_map = np.zeros_like(gt_file_np_arr, dtype=np.uint8)
    distances_map = np.zeros((gt_file_np_arr.shape[0], gt_file_np_arr.shape[1], max_label), np.float32)
    for k in range(min_label+1, max_label+1):
        kth_blob = gt_file_np_arr == k
        eroded_blob = morphology.erosion(kth_blob)
        diff_image = np.logical_and(kth_blob, np.logical_not(eroded_blob))
        twice_eroded_blob = morphology.erosion(eroded_blob)
        kth_contour = np.logical_and(eroded_blob, np.logical_not(twice_eroded_blob))
        gt_file_np_arr[diff_image] = 0
        cell_edges_map[kth_contour] = k
        pix_indices, distances = contour_to_min_dist_array(kth_contour)
        distances_map[pix_indices[:, 0], pix_indices[:, 1], k-1] = distances
        # plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.imshow(gt_file_np_arr, interpolation='none')
        # plt.subplot(1, 3, 2)
        # plt.imshow(kth_contour, interpolation='none')
        # plt.subplot(1, 3, 3)
        # plt.imshow(distances_map[:, :, k-1].reshape(512, 512),
        #            'jet', interpolation='none', alpha=0.7)
        # plt.show()

    print('ok')
    distances_map_sorted = np.sort(distances_map)
    d1 = distances_map_sorted[:, :, 0]
    d2 = distances_map_sorted[:, :, 1]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(gt_file_np_arr, interpolation='none')
    plt.subplot(1, 2, 2)
    plt.imshow(10 * np.exp(-np.square(d1 + d2) / 50))
    # plt.subplot(1, 3, 3)
    # plt.imshow(distances_map[:, :, k-1].reshape(512, 512),
    #            'jet', interpolation='none', alpha=0.7)
    plt.show()




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