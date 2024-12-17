import cv2
from matplotlib import pyplot as plt
import sys
import os
from tkinter import *
from tkinter import filedialog
import numpy as np
import math
import torch
import torch.nn as nn

from model_init import init_model
from train_utils import load_checkpoint, circulate_finetune
from data_utils import draw_gaussian, letterbox, padding_label

class Param:
    def __init__(self):
        self.drawing = False
        self.box = None
        self.last_box = False

def on_mouse(event, x, y, flags, param):
    rect = param
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Press the left mouse button to start drawing.
        rect.box = (x, y, 0, 0)
        rect.drawing = True
        rect.last_box = False
    elif event == cv2.EVENT_LBUTTONUP:  # Release the left mouse button to stop drawing.
        rect.drawing = False
        rect.last_box = True
    elif event == cv2.EVENT_MOUSEMOVE:  # Move the mouse to update the size of the rectangle.
        if rect.drawing:
            rect.box = (rect.box[0], rect.box[1], x - rect.box[0], y - rect.box[1])

def main(model, dev, base_dir):
    root = Tk()
    root.withdraw()
    File = filedialog.askopenfilename(parent=root, initialdir=base_dir,title='Choose an image.')
    root.destroy()
    (filepath, _) = os.path.split(File)

    image = cv2.imread(os.path.join(File))
    if image is None:
        print("Error reading image...")
        return
    
    cv2.namedWindow("Image",cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    h,w,_ = image.shape
    factor = 720/max(h,w)
    cv2.resizeWindow("Image", int(w*factor), int(h*factor))

    params = Param()
    cv2.setMouseCallback("Image", on_mouse, params)

    boxes = []
    stop = False
    while True:
        cv2.imshow("Image", image)
        
        if params.drawing: 
            temp = image.copy()
            cv2.rectangle(temp, (params.box[0], params.box[1]), (params.box[0] + params.box[2], params.box[1] + params.box[3]), (0, 255, 0), 2)
            cv2.imshow("Image", temp)
        elif params.last_box:  
            cv2.rectangle(image, (params.box[0], params.box[1]), (params.box[0] + params.box[2], params.box[1] + params.box[3]), (0, 255, 0), 2)
            boxes.append((params.box[0], params.box[1],params.box[0] + params.box[2], params.box[1] + params.box[3]))
            cv2.imshow("Image", image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:  # Press the Esc key or close the window to exit the loop.
            stop = True
            break
        elif key == 108 or key == 13:  # "Press the Enter key to start detecting.
            if len(boxes)!=0:
                boxes = np.array(list(set(boxes)))
                stop = False
                break

    cv2.destroyAllWindows()
    image = cv2.imread(os.path.join(File))
    
    if stop == False:
        prompt = np.zeros((image.shape[0],image.shape[1],1), dtype=np.float32)
        for box in boxes:
            x1, y1, x2, y2 = box
            h, w = y2 - y1, x2 - x1
            radius = (math.ceil(w/2),math.ceil(h/2))
            ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            prompt[:, :, 0] = draw_gaussian(prompt[:, :, 0], ct_int, radius)
        

        s = np.mean((boxes[:,2]-boxes[:,0]+boxes[:,3]-boxes[:,1])/2)
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
        prompt = np.expand_dims(cv2.resize(prompt,(int(image.shape[1]*scale_factor),int(image.shape[0]*scale_factor)),interpolation=ip),2)
        image = cv2.resize(image,(int(image.shape[1]*scale_factor),int(image.shape[0]*scale_factor)),interpolation=ip)
        boxes = np.round(np.multiply(boxes,scale_factor))

        H = int((image.shape[0] + 32 - 1) / 32) * 32
        W = int((image.shape[1] + 32 - 1) / 32) * 32
        image, prompt, _,  ratio, pad = letterbox(image, prompt, prompt, (H,W), auto=False, scaleup=True)
        _, boxes = padding_label(np.array([[0,0]]), boxes, ratio[0], ratio[1], padw=pad[0], padh=pad[1])

        image = image/255
        data = torch.from_numpy(np.transpose(np.concatenate((image,prompt),axis=2),(2,0,1))).unsqueeze(0)
        test_data = data.to(torch.float32).to(dev)
        
        with torch.no_grad():
            output_dict = model(test_data)
            output_dict['score'] = torch.sigmoid(output_dict['score'])
            avg_pooled_pred_probmap = nn.functional.avg_pool2d(output_dict['probmap'][0], 3, stride=1, padding=1)
            max_pooled_pred_probmap = nn.functional.max_pool2d(avg_pooled_pred_probmap, 3, stride=1, padding=1)
            pred_dotmap = torch.where(avg_pooled_pred_probmap==max_pooled_pred_probmap, avg_pooled_pred_probmap, torch.full_like(output_dict['probmap'][0], 0))
            pred_dotmap = pred_dotmap[0].cpu().numpy()
            pred_countmap_0 = np.where(pred_dotmap>=0.05, 1, 0)
            pred_count_0 = np.sum(pred_countmap_0)
            pred_countmap = np.where(pred_dotmap>=output_dict['score'][0][0].cpu().numpy(), 1, 0)
            pred_count = np.sum(pred_countmap)
            if pred_count_0-pred_count>300:
                pred_count,output_dict,_,_,pred_countmap = circulate_finetune(pred_count,output_dict,boxes,test_data,pred_countmap,model,dev)
        
        print('pred count:',pred_count,' pred score:',output_dict['score'][0][0].item())
        output = np.transpose(output_dict['probmap'][0].cpu().numpy(),(1,2,0))
        cmap = plt.colormaps.get_cmap('jet')
        c_output = cmap(output[:,:,0] / (output[:,:,0].max()) + 1e-14)[:, :, 0:3] * 255.0
        image = image[..., ::-1]
        c_output = c_output*0.5+image*255*0.5

        for box in boxes:
            cv2.rectangle(c_output, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        plt.imshow(c_output.astype(np.uint8))
        plt.show(block=True)



    return stop, filepath
        
if __name__ == "__main__":
    config = {
              'pre_trained':'run/model_paper/best_similarity1.pth.tar',
              'dev':torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
              }
    base_dir = sys.path[0]
    model = init_model().to(config['dev'])
    model, _ = load_checkpoint(model,os.path.join(base_dir,config['pre_trained']))
    model = model.to(config['dev'])
    while True:
        stop, filepath = main(model, config['dev'], base_dir)
        if stop:
            break
        else:
            base_dir = filepath
