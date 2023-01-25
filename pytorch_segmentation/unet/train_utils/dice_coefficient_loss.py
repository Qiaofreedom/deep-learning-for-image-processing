import torch
import torch.nn as nn


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index) # 寻找target为255的像素
        dice_target[ignore_mask] = 0 #将255全部变成0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float() 
        #构建每个类别的ground truth.在两个不同的channel上。将原始的gt转换成针对每一个类别（2个）的GT
        dice_target[ignore_mask] = ignore_index #将255的数值填充为255
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2) #将nn.functional.one_hot处理后的[N, H, W, C]转换成【N,C,H,W】


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    #x是针对某一个类别的预测矩阵，target是针对某一个类别的GT，ignore_index是哪些数值区域我们需要忽略
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0] 
    for i in range(batch_size):
        x_i = x[i].reshape(-1) #第I张图片
        t_i = target[i].reshape(-1)
        if ignore_index >= 0: 
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index) # 寻找 t_i中不为255的区域
            x_i = x_i[roi_mask] #提取预测中感兴趣的区域
            t_i = t_i[roi_mask] #提取对应target中感兴趣的区域
        inter = torch.dot(x_i, t_i) #内积操作
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size # 计算一个batch中所有图片某个类别的dice_coefficient


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon) #遍历每个类别的dice_coeff。因为one_hot已经生成了 总类别个数 的GT

    return dice / x.shape[1] #得到左右类别的 dice_coeff 均值


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1) # 在channel方向做softmax处理。计算每个像素针对每个类别的概率
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index) #得到当前batch这个数据的 dice_coeff
