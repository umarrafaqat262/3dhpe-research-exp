import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Numpy-based errors

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1), axis=1)

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1), axis=1)


# PyTorch-based errors (for losses)

def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    # print(predicted.shape, target.shape)
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    # print(predicted.shape, w.shape)
    assert w.shape[0] == predicted.shape[2] #torch.Size([24, 243, 17, 3]) torch.Size([17])
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def loss_2d_weighted(predicted, target, conf):
    assert predicted.shape == target.shape
    predicted_2d = predicted[:,:,:,:2]
    target_2d = target[:,:,:,:2]
    diff = (predicted_2d - target_2d) * conf
    return torch.mean(torch.norm(diff, dim=-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def get_limb_lens(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens

def loss_limb_var(x):
    '''
        Input: (N, T, 17, 3)
    '''
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    limb_lens = get_limb_lens(x)
    limb_lens_var = torch.var(limb_lens, dim=1)
    limb_loss_var = torch.mean(limb_lens_var)
    return limb_loss_var

def loss_limb_gt(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_lens_x = get_limb_lens(x)
    limb_lens_gt = get_limb_lens(gt) # (N, T, 16)
    return nn.L1Loss()(limb_lens_x, limb_lens_gt)

# Left/right symmetric bone pairs, as indices into get_limb_lens output (limbs_id order):
#   limbs_id = [[0,1],[1,2],[2,3],  [0,4],[4,5],[5,6],  [0,7],[7,8],[8,9],[9,10],
#               [8,11],[11,12],[12,13],  [8,14],[14,15],[15,16]]
#   idx:         0     1     2       3     4     5       6    7    8    9
#               10     11      12       13     14      15
# Right side: pelvis-hip 0, thigh 1, shin 2, clavicle 13, upperarm 14, forearm 15.
# Left  side: pelvis-hip 3, thigh 4, shin 5, clavicle 10, upperarm 11, forearm 12.
_SYM_LEFT  = [3, 4, 5, 10, 11, 12]
_SYM_RIGHT = [0, 1, 2, 13, 14, 15]

def loss_limb_symmetry(x):
    '''
    Left/right bone-length symmetry prior. A human skeleton is bilaterally symmetric
    (left femur == right femur, etc.), so the corresponding left/right bone lengths of a
    prediction should match. This is a self-consistency prior (no ground truth needed):
    when one side is occluded and predicted poorly, matching it to the visible mirror side
    regularizes the hidden limb. Addresses occlusion WITHOUT any data augmentation.
        Input: (N, T, 17, 3)  ->  scalar
    '''
    limb_lens = get_limb_lens(x)                        # (N, T, 16)
    left  = limb_lens[..., _SYM_LEFT]                   # (N, T, 6)
    right = limb_lens[..., _SYM_RIGHT]                  # (N, T, 6)
    return torch.mean(torch.abs(left - right))

def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:,1:] - predicted[:,:-1]
    velocity_target = target[:,1:] - target[:,:-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))

def loss_joint(predicted, target):
    assert predicted.shape == target.shape
    return nn.L1Loss()(predicted, target)

def get_angles(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    angle_id = [[ 0,  3],
                [ 0,  6],
                [ 3,  6],
                [ 0,  1],
                [ 1,  2],
                [ 3,  4],
                [ 4,  5],
                [ 6,  7],
                [ 7, 10],
                [ 7, 13],
                [ 8, 13],
                [10, 13],
                [ 7,  8],
                [ 8,  9],
                [10, 11],
                [11, 12],
                [13, 14],
                [14, 15] ]
    eps = 1e-7
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    angles = limbs[:,:,angle_id,:]
    angle_cos = F.cosine_similarity(angles[:,:,:,0,:], angles[:,:,:,1,:], dim=-1)
    return torch.acos(angle_cos.clamp(-1+eps, 1-eps)) 

def loss_angle(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_angles_x = get_angles(x)
    limb_angles_gt = get_angles(gt)
    return nn.L1Loss()(limb_angles_x, limb_angles_gt)

def loss_angle_velocity(x, gt):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    x_a = get_angles(x)
    gt_a = get_angles(gt)
    x_av = x_a[:,1:] - x_a[:,:-1]
    gt_av = gt_a[:,1:] - gt_a[:,:-1]
    return nn.L1Loss()(x_av, gt_av)

