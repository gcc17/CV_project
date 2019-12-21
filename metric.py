def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    #print('extract_classes',cl,n_cl)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)
 
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def dice_coeff(pred, target):
    smooth = 1.
    #print('aaaaaaaaaaaaaaaaaaaa',pred)
    #print(pred.size())
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    #return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    return (2. * intersection) / (m1.sum() + m2.sum())

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    #print('meaniu',cl,n_cl)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :]
        curr_gt_mask = gt_mask[i, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
 
    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_

def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :] = segm == c

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    #print(h_e,w_e)
    h_g, w_g = segm_size(gt_segm)
    #print(h_g,w_g)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def binary_dice3d(s,g):
    #dice score of two 3D volumes
    num=np.sum(np.multiply(s, g))
    denom=s.sum() + g.sum() 
    if denom==0:
        return 1
    else:
        return  2.0*num/denom


def sensitivity (seg,ground): 
    #computs false negative rate
    num=np.sum(np.multiply(ground, seg ))
    denom=np.sum(ground)
    #print('sen',num,denom)
    if denom==0:
        return 1
    else:
        return  num/denom

def specificity (seg,ground): 
    #computes false positive rate
    num=np.sum(np.multiply(ground==0, seg ==0))
    denom=np.sum(ground==0)
    #print('spe',num,denom)
    if denom==0:
        return 1
    else:
        return  num/denom

def iou(pred, target, n_classes = 2):
  ious = []
  num = pred.size(0)
  pred = pred.view(num,-1)
  target = target.view(num,-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      #print('qqqqqq',float(intersection))
      #print(float(max(union,1)))
      ious.append(float(intersection) / float(max(union, 1)))
  return np.array(ious[0])


def DSC_whole(pred, orig_label):
    #computes dice for the whole tumor
    return binary_dice3d(pred>0,orig_label>0)

def sensitivity_whole (seg,ground):
    return sensitivity(seg>0,ground>0)

def specificity_whole (seg,ground):
    return specificity(seg>0,ground>0)

def evaluate_segmented_volume(predicted_images,gt,predicted_images1,gt1):
    Dice_complete=DSC_whole(predicted_images,gt)

    #gt1 = gt1.to(device)
    #predicted_images1 = predicted_images1.to(device)
    #dice = float(dice_coeff(predicted_images1,gt1))

    Sensitivity_whole=sensitivity_whole(predicted_images,gt)

    Specificity_whole=specificity_whole(predicted_images,gt)

    jaccard = iou(predicted_images1,gt1,2)
    jaccard1 = mean_IU(predicted_images,gt)

    pixelacc = pixel_accuracy(predicted_images,gt)

    return Dice_complete,Sensitivity_whole,Specificity_whole,jaccard,jaccard1,pixelacc
