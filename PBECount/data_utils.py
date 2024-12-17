import numpy as np
import torch
import cv2
import random
import albumentations as A

transform = A.Compose([
        A.Blur(p=0.35),
        A.MedianBlur(p=0.35),
        A.ToGray(p=0.025),
        A.CLAHE(p=0.1)
    ])

def draw_gaussian(heatmap, center, radius, k=1):
    """
    Get a heatmap of one class
    Args:
        heatmap: The heatmap of one class(storage in single channel)
        center: The location of object center
        radius: 2D Gaussian circle radius
        k: The magnification of the Gaussian

    Returns: heatmap

    """
    diameter = 2 * max(radius[0],radius[1]) + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    gaussian = cv2.resize(gaussian,dsize=(radius[0]*2+1,radius[1]*2+1),interpolation=cv2.INTER_LINEAR)
    #print(gaussian.shape)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius[0]), min(width - x, radius[0] + 1)
    top, bottom = min(y, radius[1]), min(height - y, radius[1] + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius[1] - top:radius[1] + bottom, radius[0] - left:radius[0] + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

def gaussian2D(shape, sigma=1):
    """
    2D Gaussian function
    Args:
        shape: (diameter, diameter)
        sigma: variance

    Returns: h

    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h

def letterbox(im, prompt, probmap, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        probmap = cv2.resize(probmap, new_unpad, interpolation=cv2.INTER_LINEAR)
        prompt = cv2.resize(prompt, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    probmap = cv2.copyMakeBorder(probmap, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    prompt = cv2.copyMakeBorder(prompt, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return im, np.expand_dims(prompt,axis=2), np.expand_dims(probmap,axis=2), ratio, (dw, dh)

def padding_label(x, x1, w, h, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y1 = x1.clone() if isinstance(x1, torch.Tensor) else np.copy(x1)
    y[:, 0] = w * (x[:, 0]) + padw  # top left x
    y[:, 1] = h * (x[:, 1]) + padh  # top left y

    y1[:, 0] = w * (x1[:, 0]) + padw  
    y1[:, 1] = h * (x1[:, 1]) + padh  
    y1[:, 2] = w * (x1[:, 2]) + padw  
    y1[:, 3] = h * (x1[:, 3]) + padh  
    return y,y1

def augment_hsv(im, hgain=0.0075, sgain=0.35, vgain=0.2):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

def augment_blur(img):
    new = transform(image=img)  # transformed
    img = new['image']
    return img

def batch_augment(img, prompt, probmap, points, boxes, flipud = 0.5, fliplr = 0.5, scalep = (1,0), scaler = (0.5,0.5,0.2)):
    if max(img.shape[0]/img.shape[1],img.shape[1]/img.shape[0])<1.5:
        img, prompt, probmap, points, boxes = random_rotate(img, prompt, probmap, points, boxes)

    if random.random() < flipud:
        img = np.flipud(img).copy()
        prompt = np.flipud(prompt).copy()
        probmap = np.flipud(probmap).copy()
        points[:, 1] = img.shape[0] - points[:, 1]
        temp_coor = img.shape[0] - boxes[:,1]
        boxes[:, 1] = img.shape[0] - boxes[:, 3]
        boxes[:, 3] = temp_coor
        
    if random.random() < fliplr:
        img = np.fliplr(img).copy()
        prompt = np.fliplr(prompt).copy()
        probmap = np.fliplr(probmap).copy()
        points[:, 0] = img.shape[1] - points[:, 0]
        temp_coor = img.shape[1] - boxes[:, 0]
        boxes[:, 0] = img.shape[1] - boxes[:, 2]
        boxes[:, 2] = temp_coor
    
    if random.random() < scalep[0]:
        scale_factor = random.uniform(1-scaler[0],1+scaler[1])
        if scale_factor>=1:
            ip = cv2.INTER_CUBIC
        else:
            ip = cv2.INTER_AREA
        prompt = np.expand_dims(cv2.resize(prompt,(int(img.shape[1]*scale_factor),int(img.shape[0]*scale_factor)),interpolation=ip),2)
        probmap = cv2.resize(probmap,(int(img.shape[1]*scale_factor),int(img.shape[0]*scale_factor)),interpolation=ip)
        img = cv2.resize(img,(int(img.shape[1]*scale_factor),int(img.shape[0]*scale_factor)),interpolation=ip)
        points = np.round(np.multiply(points,scale_factor))
        boxes = np.round(np.multiply(boxes,scale_factor))
    if random.random() < scalep[1]:
        scale_factor = random.uniform(1-scaler[2],1+scaler[2])
        if scale_factor>=1:
            ip = cv2.INTER_CUBIC
        else:
            ip = cv2.INTER_AREA
        prompt = np.expand_dims(cv2.resize(prompt,(img.shape[1],int(img.shape[0]*scale_factor)),interpolation=ip),2)
        probmap = cv2.resize(probmap,(img.shape[1],int(img.shape[0]*scale_factor)),interpolation=ip)
        img = cv2.resize(img,(img.shape[1],int(img.shape[0]*scale_factor)),interpolation=ip)
        points[:, 1] = np.round(np.multiply(points[:, 1],scale_factor))
        boxes[:, 1] = np.round(np.multiply(boxes[:, 1],scale_factor))
        boxes[:, 3] = np.round(np.multiply(boxes[:, 3],scale_factor))
    
    return img, prompt, probmap, points, boxes

def random_crop(img, prompt, probmap, points, boxes, is_ori):
    if random.random() < 0.8:
        if is_ori:
            if len(points) < 50:
                area_map = np.where(probmap!=0,1,0)
            else:
                area_map = np.where(prompt!=0,1,0)
            sum_y = np.sum(area_map,1)
            sum_x = np.sum(area_map,0)
            nz_y = np.nonzero(sum_y)[0]
            nz_x = np.nonzero(sum_x)[0]

            ymin,ymax = min(nz_y),max(nz_y)
            y_range = list(range(0,ymin))+list(range(ymax+1,img.shape[0]))
            if len(y_range)==0:
                y_area = (0,img.shape[0])
            else:
                c_y = random.choice(y_range)
                if c_y < ymin:
                    y_area = (c_y,img.shape[0])
                else:
                    y_area = (0,c_y+1)
            xmin,xmax = min(nz_x),max(nz_x)
            x_range = list(range(0,xmin))+list(range(xmax+1,img.shape[1]))
            if len(x_range)==0:
                x_area = (0,img.shape[1])
            else:
                c_x = random.choice(x_range)
                if c_x < xmin:
                    x_area = (c_x,img.shape[1])
                else:
                    x_area = (0,c_x+1)
        else:
            cx = random.randint(int(min(img.shape[0]/2,img.shape[1]/2)),int(img.shape[1]))
            cy = random.randint(int(min(img.shape[0]/2,img.shape[1]/2)),int(img.shape[0]))
            mx = random.randint(0,img.shape[1]-cx)
            my = random.randint(0,img.shape[0]-cy)
            x_area,y_area = (mx,mx+cx),(my,my+cy)
        img = img[y_area[0]:y_area[1],x_area[0]:x_area[1],:]
        prompt = prompt[y_area[0]:y_area[1],x_area[0]:x_area[1],:]
        probmap = probmap[y_area[0]:y_area[1],x_area[0]:x_area[1]]
        new_points = []
        for point in points:
            if point[0]>=x_area[0] and point[0]<x_area[1] and point[1]>=y_area[0] and point[1]<y_area[1]:
                new_points.append([point[0]-x_area[0],point[1]-y_area[0]])
        points = np.array(new_points)
        if isinstance(boxes,np.ndarray):
            boxes[:,0] = boxes[:,0]-x_area[0]
            boxes[:,2] = boxes[:,2]-x_area[0]
            boxes[:,1] = boxes[:,1]-y_area[0]
            boxes[:,3] = boxes[:,3]-y_area[0]
            boxes[:,0] = np.where(boxes[:,0]>=0,boxes[:,0],np.zeros_like(boxes[:,0]))
            boxes[:,1] = np.where(boxes[:,1]>=0,boxes[:,1],np.zeros_like(boxes[:,1]))
            boxes[:,2] = np.where(boxes[:,2]<=x_area[1]-x_area[0],boxes[:,2],np.ones_like(boxes[:,2])*(x_area[1]-x_area[0]))
            boxes[:,3] = np.where(boxes[:,3]<=y_area[1]-y_area[0],boxes[:,3],np.ones_like(boxes[:,3])*(y_area[1]-y_area[0]))

        return img, prompt, probmap, points, boxes
    else:
        return img, prompt, probmap, points, boxes
    
def random_flip(img, prompt, probmap, points, flipud = 0.5, fliplr = 0.5):
    if random.random() < flipud:
        img = np.flipud(img).copy()
        prompt = np.flipud(prompt).copy()
        probmap = np.flipud(probmap).copy()
        if len(points)!=0:
            points[:, 1] = img.shape[0] - points[:, 1]
    if random.random() < fliplr:
        img = np.fliplr(img).copy()
        prompt = np.fliplr(prompt).copy()
        probmap = np.fliplr(probmap).copy()
        if len(points)!=0:
            points[:, 0] = img.shape[1] - points[:, 0]
    
    return img.copy(), prompt, probmap, points

def random_resize(img, prompt, probmap, points=[], scalep = (0.5,0.2), scaler = (0.2,0.2,0.2)):
    if random.random() < scalep[0]:
        scale_factor = random.uniform(1-scaler[0],1+scaler[1])
        if scale_factor>=1:
            ip = cv2.INTER_CUBIC
        else:
            ip = cv2.INTER_AREA
        new_prompt = np.expand_dims(cv2.resize(prompt,(int(img.shape[1]*scale_factor),int(img.shape[0]*scale_factor)),interpolation=ip),2)
        new_probmap = cv2.resize(probmap,(int(img.shape[1]*scale_factor),int(img.shape[0]*scale_factor)),interpolation=ip)
        new_img = cv2.resize(img,(int(img.shape[1]*scale_factor),int(img.shape[0]*scale_factor)),interpolation=ip)
        if len(points)!=0:
            new_points = np.round(np.multiply(points,scale_factor))
        else:
            new_points = []
    else:
        new_prompt = prompt.copy()
        new_probmap = probmap.copy()
        new_img = img.copy()
        new_points = points.copy()
    if random.random() < scalep[1]:
        scale_factor = random.uniform(1-scaler[2],1+scaler[2])
        if scale_factor>=1:
            ip = cv2.INTER_CUBIC
        else:
            ip = cv2.INTER_AREA
        new_prompt = np.expand_dims(cv2.resize(new_prompt,(new_img.shape[1],int(new_img.shape[0]*scale_factor)),interpolation=ip),2)
        new_probmap = cv2.resize(new_probmap,(new_img.shape[1],int(new_img.shape[0]*scale_factor)),interpolation=ip)
        new_img = cv2.resize(new_img,(new_img.shape[1],int(new_img.shape[0]*scale_factor)),interpolation=ip)
        if len(new_points)!=0:
            new_points = new_points.copy()
            new_points[:, 1] = np.round(np.multiply(new_points[:, 1],scale_factor))
        else:
            new_points = []
    else:
        new_prompt = new_prompt.copy()
        new_probmap = new_probmap.copy()
        new_img = new_img.copy()
        new_points = new_points.copy()
            
    return new_img, new_prompt, new_probmap, new_points

def random_rotate(img, prompt, probmap, points, boxes):
    r_factor = random.randint(0,3)
    r_img = np.rot90(img,r_factor)
    r_prompt = np.rot90(prompt,r_factor)
    r_probmap = np.rot90(probmap,r_factor)
    if r_factor == 0:
        r_points = points
        r_boxes = boxes
    elif r_factor == 1:
        r_points = np.zeros_like(points)
        r_points[:,0] = points[:,1]
        r_points[:,1] = img.shape[1]-points[:,0]
        r_boxes = np.zeros_like(boxes)
        r_boxes[:,0] = boxes[:,1]
        r_boxes[:,1] = img.shape[1]-boxes[:,2]
        r_boxes[:,2] = boxes[:,3]
        r_boxes[:,3] = img.shape[1]-boxes[:,0]
    elif r_factor == 2:
        r_points = np.zeros_like(points)
        r_points[:,0] = img.shape[1]-points[:,0]
        r_points[:,1] = img.shape[0]-points[:,1]
        r_boxes = np.zeros_like(boxes)
        r_boxes[:,0] = img.shape[1]-boxes[:,2]
        r_boxes[:,1] = img.shape[0]-boxes[:,3]
        r_boxes[:,2] = img.shape[1]-boxes[:,0]
        r_boxes[:,3] = img.shape[0]-boxes[:,1]
    elif r_factor == 3:
        r_points = np.zeros_like(points)
        r_points[:,0] = img.shape[0]-points[:,1]
        r_points[:,1] = points[:,0]
        r_boxes = np.zeros_like(boxes)
        r_boxes[:,0] = img.shape[0]-boxes[:,3]
        r_boxes[:,1] = boxes[:,0]
        r_boxes[:,2] = img.shape[0]-boxes[:,1]
        r_boxes[:,3] = boxes[:,2]

    return r_img, r_prompt, r_probmap, r_points, r_boxes

