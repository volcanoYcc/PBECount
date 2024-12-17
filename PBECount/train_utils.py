import os

import torch
import torch.nn as nn
import numpy as np
import scipy
import math

from tqdm import tqdm

from data_utils import draw_gaussian


os.environ['KMP_DUPLICATE_LIB_OK']='True'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(model, train_loader, epoch, optimizer, scheduler, dev):
    model.train()
    losses_dict = {'all':AverageMeter()}
    losses_dict['prob'] = AverageMeter()
    losses_dict['score'] = AverageMeter()

    with tqdm(train_loader) as tbar:
        for train_data, probmap, points, pointmap, _, _ in tbar:
            tbar.set_description("epoch {}".format(epoch))
            train_data = train_data.to(torch.float32).to(dev)
            probmap = probmap.to(dev)
            gt_count = points.shape[1]
            
            optimizer.zero_grad()

            output_dict = model(train_data)

            factors = torch.ones_like(probmap).to(dev)
            g = 32
            split1 = output_dict['probmap'].shape[2]//g
            split1_remain = output_dict['probmap'].shape[2]%g
            split2 = output_dict['probmap'].shape[3]//g
            split2_remain = output_dict['probmap'].shape[3]%g
            for i in range(split1):
                for j in range(split2):
                    factors[:,:,i*g:(i+1)*g,j*g:(j+1)*g] = factors[:,:,i*g:(i+1)*g,j*g:(j+1)*g] + torch.sum(pointmap[:,:,i*g:(i+1)*g,j*g:(j+1)*g])
                if split2_remain != 0:
                    factors[:,:,i*g:(i+1)*g,split2*g:] = factors[:,:,i*g:(i+1)*g,split2*g:] + torch.sum(pointmap[:,:,i*g:(i+1)*g,split2*g:])
            if split1_remain != 0:
                for j in range(split2):
                    factors[:,:,split1*g:,j*g:(j+1)*g] = factors[:,:,split1*g:,j*g:(j+1)*g] + torch.sum(pointmap[:,:,split1*g:,j*g:(j+1)*g])
            if split1_remain != 0 and split2_remain != 0:
                factors[:,:,split1*g:,split2*g:] = factors[:,:,split1*g:,split2*g:] + torch.sum(pointmap[:,:,split1*g:,split2*g:])

            loss = torch.tensor(0).to(dev)

            criterion = nn.MSELoss(reduction='none')
            loss_probmap = criterion(output_dict['probmap'].float(), probmap.float())
            loss_probmap = loss_probmap * factors
            loss_probmap = torch.sum(loss_probmap)
            losses_dict['prob'].update(loss_probmap.item(), train_data.shape[0])
            loss = loss+loss_probmap

            avg_pooled_pred_probmap = nn.functional.avg_pool2d(output_dict['probmap'][0], 3, stride=1, padding=1)
            max_pooled_pred_probmap = nn.functional.max_pool2d(avg_pooled_pred_probmap, 3, stride=1, padding=1)
            pred_dotmap = torch.where(avg_pooled_pred_probmap==max_pooled_pred_probmap, avg_pooled_pred_probmap, torch.full_like(output_dict['probmap'][0], 0))
            pred_sortmap = pred_dotmap.reshape(-1)
            pred_sortmap,_ = torch.sort(pred_sortmap,descending=True)
            nonzero_num = torch.sum((torch.where(pred_sortmap>0,1,0)))
            if nonzero_num < gt_count:
                gt_score = pred_sortmap[nonzero_num-1]/2
            else:
                gt_score = (pred_sortmap[gt_count-1]+pred_sortmap[gt_count])/2
            criterion = nn.BCEWithLogitsLoss()
            loss_score = criterion(output_dict['score'].float()[0][0], gt_score.float())
            losses_dict['score'].update(loss_score.item(), train_data.shape[0])
            loss = loss + loss_score

            losses_dict['all'].update(loss.item(), train_data.shape[0])

            loss.backward()
            optimizer.step()

            tbar_text = ''
            return_losses = {}
            for key in losses_dict.keys():
                tbar_text = tbar_text+key+':'+"{:.4f}".format(losses_dict[key].avg)+' '
                return_losses[key] = round(losses_dict[key].avg,4)

            tbar.set_postfix(inf=tbar_text)

            del train_data, probmap, pointmap, loss, factors, output_dict, loss_probmap, loss_score

    scheduler.step()

    return return_losses

def eval_one_epoch(model, val_loader, post_scores, dev):
    model.eval()
    all_gt = []

    pred_count_dict = {}
    result_dict = {}
    pred_count_dict['dyna_probmap'] = []

    if len(post_scores) != 0:
        pred_count_dict['post_probmap'] = []
        for _ in post_scores:
            pred_count_dict['post_probmap'].append([])
    
    with tqdm(val_loader) as tbar:
        with torch.no_grad():
            for test_data, _, points, _, boxes, _ in tbar:
                tbar.set_description("evaluating")
                test_data = test_data.to(torch.float32).to(dev)
                gt_count = points.shape[1]
                all_gt.append(gt_count)

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
                    pred_count,output_dict,_,_,pred_countmap = circulate_finetune(pred_count,output_dict,boxes.cpu().numpy()[0],test_data,pred_countmap,model,dev)
                pred_count_dict['dyna_probmap'].append(pred_count)
                
                if len(post_scores) != 0:
                    prompt_num = boxes.shape[1]
                    
                    if pred_count < 1000:
                        hard_img_d = False
                    else:
                        pts = np.array(list(zip(np.nonzero(pred_countmap)[1], np.nonzero(pred_countmap)[0])))
                        tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)
                        distances, locations = tree.query(pts, k=3)
                        distances = np.sort(distances[:,1])
                        mean_dis = np.mean(distances[:min(50,pred_count)])
                        if mean_dis < 7.5:
                            hard_img_d = True
                        else:
                            hard_img_d = False
                    
                    for i in range(len(post_scores)):
                        hard_img = hard_img_d
                        for box in boxes[0]:
                            x1,y1,x2,y2 = box
                            object_in_prompt = np.where((test_data[0][-1][y1:y2,x1:x2].cpu().numpy()>=post_scores[i])&(pred_countmap[y1:y2,x1:x2]==1),test_data[0][-1][y1:y2,x1:x2].cpu().numpy(),0)
                            pred_sortmap = object_in_prompt.reshape(-1)
                            pred_sortmap = abs(np.sort(-pred_sortmap))
                            nonzero_num = np.sum((np.where(pred_sortmap>0,1,0)))
                            if nonzero_num > 1:
                                hard_img = True
                                break
                        
                        if hard_img:
                            t1 = np.sum(output_dict['probmap'][0].cpu().numpy())
                            t2 = np.sum(np.where(test_data[0][-1].cpu().numpy()>=post_scores[i],output_dict['probmap'][0].cpu().numpy(),0))
                            if t2 != 0:
                                pred_count_dict['post_probmap'][i].append(t1/(t2/prompt_num))
                            else:
                                pred_count_dict['post_probmap'][i].append(pred_count)
                        else:
                            pred_count_dict['post_probmap'][i].append(pred_count)
                        

            del test_data, output_dict

    result_dict['all']={'maes':[],'rmses':[]}
    for key,value in pred_count_dict.items():
        if key == 'probmap' or key == 'post_probmap':
            maes = []
            rmses = []
            for all_pred in pred_count_dict[key]:
                mae = 0
                rmse = 0
                for i in range(len(all_pred)):
                    et_count = all_pred[i]
                    gt_count = all_gt[i]

                    mae += abs(gt_count-et_count)
                    rmse += ((gt_count-et_count)*(gt_count-et_count))

                mae = mae/len(all_pred)
                rmse = np.sqrt(rmse/(len(all_pred)))

                maes.append(round(mae,4))
                rmses.append(round(rmse,4))
        else:
            mae = 0
            rmse = 0
            all_pred = pred_count_dict[key]
            for i in range(len(all_pred)):
                et_count = all_pred[i]
                gt_count = all_gt[i]

                mae += abs(gt_count-et_count)
                rmse += ((gt_count-et_count)*(gt_count-et_count))

            mae = mae/len(all_pred)
            rmse = np.sqrt(rmse/(len(all_pred)))

            maes=[round(mae,4)]
            rmses=[round(rmse,4)]
        result_dict[key] = {'maes':maes,'rmses':rmses}

        result_dict['all']['maes'] = result_dict['all']['maes']+maes
        result_dict['all']['rmses'] = result_dict['all']['rmses']+rmses

    return result_dict

def multi_save(best_mae_list,best_epoch_list,mae,epoch,base_dir,train_name,train_type,data_type,model,config):
    if len(best_mae_list) == 0:
        best_mae_list.append(mae)
        best_epoch_list.append(epoch)
        save_checkpoint(model,{'epoch':epoch,'base_root': os.path.join(base_dir,'run','train',train_name,train_type)},'best_mae_'+data_type+'_0')
    else:    
        for i in range(len(best_mae_list)):
            if mae < best_mae_list[i]:
                if len(best_mae_list) < config['save_num']:
                    for j in range(len(best_mae_list)-1,i-1,-1):
                        os.rename(os.path.join(base_dir,'run','train',train_name,train_type,'best_mae_')+data_type+'_'+str(j)+'.pth.tar',
                                os.path.join(base_dir,'run','train',train_name,train_type,'best_mae_')+data_type+'_'+str(j+1)+'.pth.tar')
                    best_mae_list.insert(i,mae)
                    best_epoch_list.insert(i,epoch)
                    save_checkpoint(model,{'epoch':epoch,'base_root': os.path.join(base_dir,'run','train',train_name,train_type)},'best_mae_'+data_type+'_'+str(i))
                    break
                else:
                    os.remove(os.path.join(base_dir,'run','train',train_name,train_type,'best_mae_')+data_type+'_'+str(len(best_mae_list)-1)+'.pth.tar')
                    best_mae_list.pop()
                    best_epoch_list.pop()
                    for j in range(len(best_mae_list)-1,i-1,-1):
                        os.rename(os.path.join(base_dir,'run','train',train_name,train_type,'best_mae_')+data_type+'_'+str(j)+'.pth.tar',
                                os.path.join(base_dir,'run','train',train_name,train_type,'best_mae_')+data_type+'_'+str(j+1)+'.pth.tar')
                    best_mae_list.insert(i,mae)
                    best_epoch_list.insert(i,epoch)
                    save_checkpoint(model,{'epoch':epoch,'base_root': os.path.join(base_dir,'run','train',train_name,train_type)},'best_mae_'+data_type+'_'+str(i))
                    break
            elif len(best_mae_list) < config['save_num'] and i == len(best_mae_list)-1:
                best_mae_list.append(mae)
                best_epoch_list.append(epoch)
                save_checkpoint(model,{'epoch':epoch,'base_root': os.path.join(base_dir,'run','train',train_name,train_type)},'best_mae_'+data_type+'_'+str(i+1))
                break
    return best_mae_list, best_epoch_list

def save_checkpoint(model, info, name = 'test'):
    state = {
            'epoch': info['epoch'],
            'state_dict': model.state_dict(),
            }
    torch.save(state, os.path.join(info['base_root'], name+'.pth.tar'))

def load_checkpoint(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    epoch = state_dict['epoch']+1
    return model,epoch

def circulate_finetune(pred_count,output_dict,boxes,test_data,pred_countmap,model,dev):
    max_pred_count = pred_count
    stop_iter_num = 1
    all_new_prompt_list = []
    out_boxes = boxes.copy()
    while True:
        prompt_list = []
        ori_prompt_list = []
        for box in boxes:
            prompt_list.append([(box[0]+box[2])/2,(box[1]+box[3])/2])
            ori_prompt_list.append([(box[0]+box[2])/2,(box[1]+box[3])/2])
        if all_new_prompt_list!=[]:
            for box in all_new_prompt_list:
                prompt_list.append([(box[0]+box[2])/2,(box[1]+box[3])/2])
        prompt_list = np.array(prompt_list)
        ori_prompt_list = np.array(ori_prompt_list)

        pts = np.array(list(zip(np.nonzero(pred_countmap)[1], np.nonzero(pred_countmap)[0])))
        if pts.shape[0] == 0:
            return max_pred_count,output_dict,out_boxes,test_data,pred_countmap
        Dps = np.zeros_like(np.nonzero(pred_countmap)[0].shape[0])
        for i in range(prompt_list.shape[0]):
            box = prompt_list[i]
            Dp = np.sqrt(((pts - box) ** 2).sum(axis=1))  # distances from p
            if np.sum(Dps) == 0:
                Dps = Dp
            else:
                Dps = np.minimum(Dp,Dps)
        index = np.argmax(Dps)

        new_prompt_center = pts[index]

        for i in range(ori_prompt_list.shape[0]):
            if i == 0:
                nearest_dis = np.sqrt(((new_prompt_center - ori_prompt_list[i]) ** 2).sum())
                nearest_prompt_id = i
            else:
                temp_dis = np.sqrt(((new_prompt_center - ori_prompt_list[i]) ** 2).sum())
                if temp_dis < nearest_dis:
                    nearest_dis = temp_dis
                    nearest_prompt_id = i

        nearest_prompt_center = ori_prompt_list[nearest_prompt_id]
        nearest_prompt_area = abs((boxes[nearest_prompt_id][2]-boxes[nearest_prompt_id][0])*(boxes[nearest_prompt_id][3]-boxes[nearest_prompt_id][1]))
        
        mean_prompt_center = ori_prompt_list.sum(axis=0)/ori_prompt_list.shape[0]
        mean_prompt_area = 0
        for box in boxes:
            mean_prompt_area = mean_prompt_area+abs((box[2]-box[0])*(box[3]-box[1]))
        mean_prompt_area = mean_prompt_area/boxes.shape[0]
        mean_dis = np.sqrt(((new_prompt_center - mean_prompt_center) ** 2).sum())
        
        s1 = np.mean((boxes[:,2]-boxes[:,0])/1)
        s2 = np.mean((boxes[:,3]-boxes[:,1])/1)
        
        s1_ = s1*math.sqrt((nearest_prompt_area/mean_prompt_area)*(mean_dis/nearest_dis))
        s2_ = s2*math.sqrt((nearest_prompt_area/mean_prompt_area)*(mean_dis/nearest_dis))
        if s1_<s1:
            s1 = s1_
            s2 = s2_

        all_new_prompt_list.append([max(int(new_prompt_center[0]-s1/2),0),max(int(new_prompt_center[1]-s2/2),0),min(int(new_prompt_center[0]+s1/2),test_data.shape[3]-1),min(int(new_prompt_center[1]+s2/2),test_data.shape[2]-1)])

        temp_new_boxes = np.concatenate((boxes,np.array(all_new_prompt_list)),axis=0)
        temp_new_boxes = torch.tensor(temp_new_boxes)

        temp_new_prompt = np.zeros((test_data.shape[2],test_data.shape[3],1), dtype=np.float32)
        for box in temp_new_boxes.cpu().numpy():
            x1, y1, x2, y2 = box
            h, w = y2 - y1, x2 - x1
            radius = (math.ceil(w/2),math.ceil(h/2))
            ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            temp_new_prompt[:, :, 0] = draw_gaussian(temp_new_prompt[:, :, 0], ct_int, radius)
        
        temp_new_test_data = test_data.clone()
        temp_new_test_data[:,-1,:,:] = torch.from_numpy(np.transpose(temp_new_prompt,(2,0,1))).to(dev)

        temp_new_output_dict = model(temp_new_test_data)
        temp_new_avg_pooled_pred_probmap = nn.functional.avg_pool2d(temp_new_output_dict['probmap'][0], 3, stride=1, padding=1)
        temp_new_max_pooled_pred_probmap = nn.functional.max_pool2d(temp_new_avg_pooled_pred_probmap, 3, stride=1, padding=1)
        temp_new_pred_dotmap = torch.where(temp_new_avg_pooled_pred_probmap==temp_new_max_pooled_pred_probmap, temp_new_avg_pooled_pred_probmap, torch.full_like(temp_new_output_dict['probmap'][0], 0))
        temp_new_pred_dotmap = temp_new_pred_dotmap[0].cpu().numpy()
        temp_new_pred_countmap = np.where(temp_new_pred_dotmap>=output_dict['score'][0][0].cpu().numpy(), 1, 0)
        temp_new_pred_count = np.sum(temp_new_pred_countmap)

        if temp_new_pred_count>max_pred_count:
            max_pred_count = temp_new_pred_count
            test_data = temp_new_test_data
            output_dict['probmap'] = temp_new_output_dict['probmap']
            pred_countmap = temp_new_pred_countmap
            out_boxes = np.concatenate((boxes,np.array(all_new_prompt_list)),axis=0)
            stop_iter_num = 1
        else:
            if stop_iter_num < 7:
                stop_iter_num = stop_iter_num+1
            else:
                return max_pred_count,output_dict,out_boxes,test_data,pred_countmap

