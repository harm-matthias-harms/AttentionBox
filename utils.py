'''
Modified version of the original code from Hu et al.

@author Hu et al.
@author Christian Wilms
@author Harm Matthias Harms
@date 11/03/19
'''

from __future__ import division
import numpy as np

from alchemy.utils.image import resize_blob
from alchemy.utils.mask import crop


def transplant(new_net, net, suffix=''):
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat

def expand_score(new_net, new_layer, net, layer):
    old_cl = net.params[layer][0].num
    new_net.params[new_layer][0].data[:old_cl][...] = net.params[layer][0].data
    new_net.params[new_layer][1].data[0,0,0,:old_cl][...] = net.params[layer][1].data

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def interp(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

# generate masks from an image with specified net
# :param net:           caffe net
# :param input:         input image blob ([1, 3, h, w])
# :param config:        other parameters
# :param dest_shape:    resize masks if specified
# :param image:         visualize masks if specified
# :return masks:        masks ([num, h, w])
def gen_masks_new(net, input, config, dest_shape=None, image=None):
    net.blobs['data'].reshape(*input.shape)
    net.blobs['data'].data[...] = input

    net.forward()

    ih, iw = input.shape[2:]
    if dest_shape != None:
        oh, ow = dest_shape
    else:
        oh, ow = ih, iw
    ih, iw, oh, ow = int(ih), int(iw), int(oh), int(ow)

    if hasattr(config, 'TEST_RFs'):
        ratios = config.TEST_RFs
    else:
        ratios = config.RFs
        
    ratios = [int(ratio) for ratio in ratios]
    scales = []
    for rf in ratios:
        scales.append(((ih//rf)+(ih%rf>0), (iw//rf)+(iw%rf>0)))

    _ = 0 
    dynamicK = 1000
    if len(net.blobs['objn'].data) < dynamicK:
        dynamicK = len(net.blobs['objn'].data)
    #determine dynamically how many windows are sampled at test time
    #might be less than 1000
    ret_masks = np.zeros((dynamicK, oh, ow), dtype=np.uint8) 
    ret_scores = np.zeros((dynamicK))
    
    for topk in net.blobs['top_k'].data[:,0,0,0][:dynamicK]:
        if topk < net.blobs['obj_indices'].data[...].shape[0]:
            bid = net.blobs['obj_indices'].data[int(topk),0,0,0]
            bid = int(bid)
            score = float(net.blobs['objn'].data[int(topk)])
            scale_idx = 0
            h, w = scales[scale_idx]
            ceiling = (h + 1) * (w + 1) 
            while bid >= ceiling:
                bid -= ceiling
                scale_idx += 1
                try:
                    h, w = scales[scale_idx]
                except Exception as e:
                    raise e
                ceiling = (h + 1) * (w + 1)
    
            stride = ratios[scale_idx]
            x = bid // (w + 1) 
            y = bid % (w + 1)
    
            xb, xe = (x - config.SLIDING_WINDOW_SIZE//2) * stride, (x + config.SLIDING_WINDOW_SIZE//2) * stride 
            yb, ye = (y - config.SLIDING_WINDOW_SIZE//2) * stride, (y + config.SLIDING_WINDOW_SIZE//2) * stride
            xb, xe, yb, ye = int(round(1.0*xb*oh/ih)), int(round(1.0*xe*oh/ih)), int(round(1.0*yb*ow/iw)), int(round(1.0*ye*ow/iw)) 
            size = xe - xb, ye - yb
            rel_x, rel_y, rel_w, rel_h = net.blobs['bbox_pred'].data[_]
            box_x, box_y, box_w, box_h = recalc_relative_values(rel_x, rel_y, rel_w, rel_h)
            mask = np.zeros((160, 160))
            
            if not (box_x == 0 and box_y == 0 and box_w == 0 and box_h == 0):
                # plus one because otherwise the box is one to small in size
                mask[int(box_x):int(box_x + box_w + 1), int(box_y):int(box_y + box_h + 1)] = 1
            mask = resize_blob(mask, size) 
            mask = crop(mask, (xb, xe, yb, ye), (oh, ow))[0]
            xb = max(0, xb)
            xe = min(oh, xe)
            yb = max(0, yb)
            ye = min(ow, ye)
            #mask[mask < 0.2] = 0
            #mask[mask >= 0.2] = 1
            ret_masks[_, xb:xe, yb:ye] = mask
    
            ret_scores[_] = score
            _ += 1
        
    return ret_masks, ret_scores

#returns bbox_x, bbox_y, width, height
def recalc_relative_values(rel_x, rel_y, rel_w, rel_h):
    target_dx = rel_x * 0.074 - 0.021
    target_dy = rel_y * 0.075 - 0.009
    targets_dw = rel_w * 0.447 - 0.832
    targets_dh = rel_h * 0.5 - 0.958

    # calce width and height, it's needed for the left top point of the bbox
    #rel_w = np.log((rmax - rmin) / 95.4) / 0.06
    width = np.exp(targets_dw) * 160
    height = np.exp(targets_dh) * 160
    #recalc point from center
    # rel_x = ((center_x - 79.5) / 39.75) / 0.1
    center_x = (target_dx * 160) + 79.5
    center_y = (target_dy * 160) + 79.5
    # calc x and y
    bbox_x = center_x - (width / float(2))
    bbox_y = center_y - (height / float(2))
    return bbox_x, bbox_y, width, height