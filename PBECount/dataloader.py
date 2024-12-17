from torch.utils.data import Dataset
import os
import sys
import json
import cv2
import random
import torch
import numpy as np
import math
import h5py
import matplotlib.pyplot as plt

from data_utils import draw_gaussian, letterbox, padding_label, batch_augment, augment_hsv, random_rotate, random_crop, augment_blur

class CA_Dataset(Dataset):
    def __init__(self, data_dir, split = 'train', train = True, crop_aug = True):
        self.split = split
        self.train = train
        self.crop_aug = crop_aug
        self.anno_file = os.path.join(data_dir, 'annotation_FSC147_384.json')
        self.data_split_file = os.path.join(data_dir, 'Train_Test_Val_FSC_147.json')
        self.data_class_file = os.path.join(data_dir, 'ImageClasses_FSC147.txt')
        self.im_dir = os.path.join(data_dir, 'images_384_VarV2')
        self.probmap_dir = os.path.join(data_dir, 'images_384_VarV2_probmap')

        with open(self.anno_file) as f:
            self.annotations = json.load(f)
        with open(self.data_split_file) as f:
            data_split = json.load(f)
        with open(self.data_class_file) as f:
            self.im_classes = {}
            lines = f.readlines()
            for line in lines:
                name, cate = line.replace('\n','').split('\t')
                self.im_classes[name] = cate

        self.names = data_split[split]

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        if self.train == False:
            img, prompt, probmap, points, boxes, img_name = self.load_img(index)
            t_boxes = boxes
            s = np.mean((t_boxes[:,2]-t_boxes[:,0]+t_boxes[:,3]-t_boxes[:,1])/2)
            if s < 15:
                scale_factor = 15/1.5/s
            elif s > 80:
                scale_factor = 80/1.5/s
            else:
                scale_factor = 1
            scale_factor = scale_factor*1.5
            if scale_factor>=1:
                ip = cv2.INTER_CUBIC
            else:
                ip = cv2.INTER_AREA
            prompt = np.expand_dims(cv2.resize(prompt,(int(img.shape[1]*scale_factor),int(img.shape[0]*scale_factor)),interpolation=ip),2)
            probmap = cv2.resize(probmap,(int(img.shape[1]*scale_factor),int(img.shape[0]*scale_factor)),interpolation=ip)
            img = cv2.resize(img,(int(img.shape[1]*scale_factor),int(img.shape[0]*scale_factor)),interpolation=ip)
            points = np.round(np.multiply(points,scale_factor))
            boxes = np.round(np.multiply(boxes,scale_factor))

        elif self.train == True:
            img, prompt, probmap, points, boxes, img_name = self.load_augmented_img(index)

        H = int((img.shape[0] + 32 - 1) / 32) * 32
        W = int((img.shape[1] + 32 - 1) / 32) * 32
        img, prompt, probmap, ratio, pad = letterbox(img, prompt, probmap, (H,W), auto=False, scaleup=True)
        points, boxes = padding_label(points, boxes, ratio[0], ratio[1], padw=pad[0], padh=pad[1])

        img = img/255
        train_data = np.transpose(np.concatenate((img,prompt),axis=2),(2,0,1))
        probmap = np.transpose(probmap,(2,0,1))

        pointmap = np.zeros_like(probmap)
        for point in points:
            if point[1] >= pointmap.shape[1]:
                point[1] = pointmap.shape[1]-1
            if point[0] >= pointmap.shape[2]:
                point[0] = pointmap.shape[2]-1
            pointmap[0,int(point[1])-1,int(point[0])-1] = 1

        return train_data, probmap, points, pointmap, boxes.astype(np.int64), img_name
    
    def load_img(self, index):
        img_name = self.names[index]
        img = cv2.imread(os.path.join(self.im_dir,img_name))
        prob_file = h5py.File(os.path.join(self.probmap_dir,img_name).replace('.jpg','_probmap.h5'), 'r')
        probmap = np.asarray(prob_file['density'])
        ann = self.annotations[img_name]
        prompt = np.zeros((img.shape[0],img.shape[1],1), dtype=np.float32)

        boxes = ann['box_examples_coordinates']
        if self.train and random.random()>0.75:
            boxes = random.sample(boxes,len(boxes)-1)
            if len(boxes)>1 and self.train and random.random()>0.5:
                boxes = random.sample(boxes,len(boxes)-1)

        new_boxes = []
        for bbox in boxes:
            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[2][0], bbox[2][1]
            if self.train and random.random()>0.5:
                w = abs(x2-x1)
                h = abs(y2-y1)
                x1 = x1+random.uniform(-w*0.1,w*0.1)
                x2 = x2+random.uniform(-w*0.1,w*0.1)
                y1 = y1+random.uniform(-h*0.1,h*0.1)
                y2 = y2+random.uniform(-h*0.1,h*0.1)
            new_boxes.append([x1,y1,x2,y2])
            h, w = y2 - y1, x2 - x1
            radius = (math.ceil(w/2),math.ceil(h/2))
            ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            prompt[:, :, 0] = draw_gaussian(prompt[:, :, 0], ct_int, radius)

        points = np.array(ann['points'])

        return img, prompt, probmap, points, np.array(new_boxes), img_name
    
    def load_augmented_img(self, index):
        data_list = []
        img, prompt, probmap, points, boxes, img_name = self.load_img(index)
        ori_name = img_name
        img, prompt, probmap, points, boxes = batch_augment(img, prompt, probmap, points, boxes)
        img, prompt, probmap, points, boxes = random_crop(img, prompt, probmap, points, boxes, True)
        data_list.append({'img':img,'prompt':prompt,'probmap':probmap,'points':points,'boxes':boxes,'img_name':img_name,'is_ori':True,'have_prompt':True})
        if random.random() > 0.2:
            ori_class_name = self.im_classes[img_name]
            if random.random() > 0.5:
                num_iter = 1
                while True:
                    while True:
                        new_index = random.randint(0,len(self.names)-1)
                        if ori_class_name != self.im_classes[self.names[new_index]]:
                            break
                    img, prompt, probmap, points, boxes, img_name = self.load_img(new_index)
                    img, prompt, probmap, points, boxes = batch_augment(img, prompt, probmap, points, boxes)
                    img, prompt, probmap, points, boxes = random_crop(img, prompt, probmap, points, None, False)
                    th1,tw1,th2,tw2 = data_list[0]['img'].shape[0],data_list[0]['img'].shape[1],img.shape[0],img.shape[1]
                    if min((th1+th2)*max(tw1,tw2),max(th1,th2)*(tw1+tw2)) < 1000*1600:
                        break
                    else:
                        if num_iter%5==0:
                            ip = cv2.INTER_AREA
                            scale_factor = 0.8
                            data_list[0]['prompt'] = np.expand_dims(cv2.resize(data_list[0]['prompt'],(int(data_list[0]['prompt'].shape[1]*scale_factor),int(data_list[0]['prompt'].shape[0]*scale_factor)),interpolation=ip),2)
                            data_list[0]['probmap'] = cv2.resize(data_list[0]['probmap'],(int(data_list[0]['probmap'].shape[1]*scale_factor),int(data_list[0]['probmap'].shape[0]*scale_factor)),interpolation=ip)
                            data_list[0]['img'] = cv2.resize(data_list[0]['img'],(int(data_list[0]['img'].shape[1]*scale_factor),int(data_list[0]['img'].shape[0]*scale_factor)),interpolation=ip)
                            data_list[0]['points'] = np.round(np.multiply(data_list[0]['points'],scale_factor))
                    num_iter = num_iter+1
                data_list.append({'img':img,'prompt':prompt,'probmap':probmap,'points':points,'boxes':boxes,'img_name':img_name,'is_ori':False})
            else:
                num_iter = 1
                while True:
                    img, prompt, probmap, points, boxes, img_name = self.load_img(index)
                    img, prompt, probmap, points, boxes = batch_augment(img, prompt, probmap, points, boxes)
                    img, prompt, probmap, points, boxes = random_crop(img, prompt, probmap, points, None, False)
                    th1,tw1,th2,tw2 = data_list[0]['img'].shape[0],data_list[0]['img'].shape[1],img.shape[0],img.shape[1]
                    if min((th1+th2)*max(tw1,tw2),max(th1,th2)*(tw1+tw2)) < 1000*1600:
                        break
                    else:
                        if num_iter%5==0:
                            ip = cv2.INTER_AREA
                            scale_factor = 0.8
                            data_list[0]['prompt'] = np.expand_dims(cv2.resize(data_list[0]['prompt'],(int(data_list[0]['prompt'].shape[1]*scale_factor),int(data_list[0]['prompt'].shape[0]*scale_factor)),interpolation=ip),2)
                            data_list[0]['probmap'] = cv2.resize(data_list[0]['probmap'],(int(data_list[0]['probmap'].shape[1]*scale_factor),int(data_list[0]['probmap'].shape[0]*scale_factor)),interpolation=ip)
                            data_list[0]['img'] = cv2.resize(data_list[0]['img'],(int(data_list[0]['img'].shape[1]*scale_factor),int(data_list[0]['img'].shape[0]*scale_factor)),interpolation=ip)
                            data_list[0]['points'] = np.round(np.multiply(data_list[0]['points'],scale_factor))
                    num_iter = num_iter+1
                data_list.append({'img':img,'prompt':prompt,'probmap':probmap,'points':points,'boxes':boxes,'img_name':img_name,'is_ori':True,'have_prompt':False})
            random.shuffle(data_list)

            h1,w1 = data_list[0]['img'].shape[:-1]
            h2,w2 = data_list[1]['img'].shape[:-1]
            if max(h1,h2)*(w1+w2)<(h1+h2)*max(w1,w2):
                new_img = np.ones((max(h1,h2),w1+w2,3), dtype=np.float32)*144
                new_prompt = np.zeros((max(h1,h2),w1+w2,1), dtype=np.float32)
                new_probmap = np.zeros((max(h1,h2),w1+w2), dtype=np.float32)
                new_points = []
                new_boxes = []
                a1 = int((max(h1,h2)-h1)/2)
                a2 = int((max(h1,h2)-h2)/2)
                new_img[a1:a1+h1,:w1,:] = data_list[0]['img']
                new_img[a2:a2+h2,w1:,:] = data_list[1]['img']
                if data_list[0]['is_ori']:
                    if data_list[0]['have_prompt'] == True:
                        new_prompt[a1:a1+h1,:w1,:] = data_list[0]['prompt']
                        for box in data_list[0]['boxes']:
                            new_boxes.append([box[0],box[1]+a1,box[2],box[3]+a1])
                    new_probmap[a1:a1+h1,:w1] = data_list[0]['probmap']
                    for point in data_list[0]['points']:
                        new_points.append([point[0],point[1]+a1])
                if data_list[1]['is_ori']:
                    if data_list[1]['have_prompt'] == True:
                        new_prompt[a2:a2+h2,w1:,:] = data_list[1]['prompt']
                        for box in data_list[1]['boxes']:
                            new_boxes.append([box[0]+w1,box[1]+a2,box[2]+w1,box[3]+a2])
                    new_probmap[a2:a2+h2,w1:] = data_list[1]['probmap']
                    for point in data_list[1]['points']:
                        new_points.append([point[0]+w1,point[1]+a2])
            else:
                new_img = np.ones((h1+h2,max(w1,w2),3), dtype=np.float32)*144
                new_prompt = np.zeros((h1+h2,max(w1,w2),1), dtype=np.float32)
                new_probmap = np.zeros((h1+h2,max(w1,w2)), dtype=np.float32)
                new_points = []
                new_boxes = []
                a1 = int((max(w1,w2)-w1)/2)
                a2 = int((max(w1,w2)-w2)/2)
                new_img[:h1,a1:a1+w1,:] = data_list[0]['img']
                new_img[h1:,a2:a2+w2,:] = data_list[1]['img']
                if data_list[0]['is_ori']:
                    if data_list[0]['have_prompt'] == True:
                        new_prompt[:h1,a1:a1+w1,:] = data_list[0]['prompt']
                        for box in data_list[0]['boxes']:
                            new_boxes.append([box[0]+a1,box[1],box[2]+a1,box[3]])
                    new_probmap[:h1,a1:a1+w1] = data_list[0]['probmap']
                    for point in data_list[0]['points']:
                        new_points.append([point[0]+a1,point[1]])
                if data_list[1]['is_ori']:
                    if data_list[1]['have_prompt'] == True:
                        new_prompt[h1:,a2:a2+w2,:] = data_list[1]['prompt']
                        for box in data_list[1]['boxes']:
                            new_boxes.append([box[0]+a2,box[1]+h1,box[2]+a2,box[3]+h1])
                    new_probmap[h1:,a2:a2+w2] = data_list[1]['probmap']
                    for point in data_list[1]['points']:
                        new_points.append([point[0]+a2,point[1]+h1])
            new_points = np.array(new_points)
            new_boxes = np.array(new_boxes)
            img, prompt, probmap, points, boxes = new_img, new_prompt, new_probmap, new_points, new_boxes

        img = img.astype(np.uint8).copy()
        augment_hsv(img)
        if self.crop_aug:
            if random.random()<0.75:
                cx = random.randint(int(img.shape[1]/4),int(img.shape[1]/1.5))
                cy = random.randint(int(img.shape[0]/4),int(img.shape[0]/1.5))
                mx = random.randint(0,img.shape[1]-cx)
                my = random.randint(0,img.shape[0]-cy)
                augment_hsv(img[my:my+cy,mx:mx+cx,:],hgain=0.015, sgain=0.7, vgain=0.4)
            if random.random()<0.75:
                cx = random.randint(int(img.shape[1]/4),int(img.shape[1]/1.5))
                cy = random.randint(int(img.shape[0]/4),int(img.shape[0]/1.5))
                mx = random.randint(0,img.shape[1]-cx)
                my = random.randint(0,img.shape[0]-cy)
                augment_hsv(img[my:my+cy,mx:mx+cx,:],hgain=0.015, sgain=0.7, vgain=0.4)
            if random.random()<0.75:
                cx = random.randint(int(img.shape[1]/3),int(img.shape[1]/1.5))
                cy = random.randint(int(img.shape[0]/3),int(img.shape[0]/1.5))
                mx = random.randint(0,img.shape[1]-cx)
                my = random.randint(0,img.shape[0]-cy)
                img[my:my+cy,mx:mx+cx,:] = augment_blur(img[my:my+cy,mx:mx+cx,:])
            if random.random()<0.75:
                cx = random.randint(int(img.shape[1]/3),int(img.shape[1]/1.5))
                cy = random.randint(int(img.shape[0]/3),int(img.shape[0]/1.5))
                mx = random.randint(0,img.shape[1]-cx)
                my = random.randint(0,img.shape[0]-cy)
                img[my:my+cy,mx:mx+cx,:] = augment_blur(img[my:my+cy,mx:mx+cx,:])
        img, prompt, probmap, points, boxes = random_rotate(img, prompt, probmap, points, boxes)
        return img, prompt, probmap, points, boxes, ori_name
    
if __name__ == '__main__':
    base_dir = os.path.dirname(sys.path[0])
    print(base_dir)
    data_dir = os.path.join(base_dir, 'FSC147_384_V2')
    dataset = CA_Dataset(data_dir, split = 'train', train = True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, (train_data, probmap, points, pointmap, boxes, img_name) in enumerate(data_loader):
        print(img_name,train_data.shape,probmap.shape,pointmap.shape,points.shape,torch.sum(pointmap),boxes)
        plt.subplot(221)
        img_to_draw = np.transpose(train_data[0][:-1].cpu().numpy()*255,(1,2,0)).astype(int).copy()
        points = points[0].cpu().numpy()
        for i in range(points.shape[0]):
            cv2.circle(img_to_draw, (int(points[i][0]),int(points[i][1])), 5, (0,0,255), -1)
        boxes = boxes[0].cpu().numpy()
        for box in boxes:
            cv2.rectangle(img_to_draw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        plt.imshow(img_to_draw[...,::-1])
        plt.subplot(222)
        prompt_to_draw = train_data[0][-1].cpu().numpy()*255
        plt.imshow(prompt_to_draw)
        plt.subplot(223)
        probmap_to_draw = probmap[0][0].cpu().numpy()
        plt.imshow(probmap_to_draw)
        plt.subplot(224)
        pointmap_to_draw = pointmap[0][0].cpu().numpy()
        plt.imshow(pointmap_to_draw)
        plt.show(block=True)
        if i == 20:
            break