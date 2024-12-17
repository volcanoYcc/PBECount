import sys
import json
import os
import scipy
import scipy.spatial
from matplotlib import pyplot as plt
import numpy as np
import h5py
import math
    
def gaussian_filter_prob(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = math.sqrt(distances[i][1])
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        filter = scipy.ndimage.gaussian_filter(pt2d, sigma, mode='constant')
        peak = filter[pt[1]][pt[0]]
        density_new = filter / float(peak)
        density = np.maximum(density_new, density)
    print('done.')
    return density

if __name__ == '__main__':
    base_dir = sys.path[0]
    data_dir = os.path.join(base_dir,'images_384_VarV2')
    anno_file = os.path.join(base_dir, 'annotation_FSC147_384.json')
    names = os.listdir(data_dir)
    with open(anno_file) as f:
        annotations = json.load(f)

    for name in names:
        img_path = os.path.join(data_dir,name)
        print(img_path)

        img= plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1]))

        ann = annotations[name]
        points = ann['points']

        for i in range(0,len(points)):
            if int(points[i][1])<img.shape[0] and int(points[i][0])<img.shape[1]:
                k[int(points[i][1]),int(points[i][0])]=1

        k1 = gaussian_filter_prob(k)
        with h5py.File(img_path.replace('images_384_VarV2','images_384_VarV2_probmap').replace('.jpg','_probmap.h5'), 'w') as hf:
            hf['density'] = k1