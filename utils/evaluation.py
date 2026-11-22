"""Evaluates the performance of a model"""
import logging
import math
import torch
import torch.nn.functional as F
import torchvision
import os
from PIL import Image
#我加的
import cv2
import numpy
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchmetrics import JaccardIndex, F1Score, Accuracy, Precision, Recall
from tqdm import tqdm

from utils.cityscapes_loader import decode_segmap as decode_segmap_cityscapes
from utils.utils import diffuse, denoise_scale
from utils.uavid_loader import decode_segmap as decode_segmap_uavid
from utils.vaihingen_buildings_loader import decode_segmap as decode_segmap_vaihingen
from utils.inria_loader import decode_segmap as decode_segmap_inria
# from utils.whu_loader import decode_segmap as decode_segmap_whu
def segmentation_cross_entropy(predicted_segmentation, target_segmentation):
    """Returns Cross Entropy Loss"""
    weights = torch.tensor([1.79, 1.0], dtype=torch.float32).to(target_segmentation.device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='sum')
    loss = criterion(predicted_segmentation, target_segmentation)
    return loss

def noise_mse(noise_predicted, noise_target):
    """Returns MSE Loss"""
    criterion = torch.nn.MSELoss(reduction='mean')
    loss = criterion(noise_predicted, noise_target)
    return loss
def noise_bce(noise_predicted, noise_target):
    """Returns BCE Loss"""
    criterion = torch.nn.BCELoss(reduction='mean')
    loss = criterion(noise_predicted, noise_target)
    return loss
def compute_total_loss(segmentation_cross_entropy):
    """Returns total loss"""
    total_loss =  (1 * segmentation_cross_entropy)
    return total_loss

class JointEdgeMaskLoss(nn.Module):
    def __init__(self, edge_weight=0.5):
        """
        edge_weight: 边缘损失在总损失中的比重。一般设为 0.5 到 1.0 之间。
        """
        super().__init__()
        self.edge_weight = edge_weight
        # 注意：这里要求网络的输出不要经过 Sigmoid，直接输出 Logits
        self.bce_loss = nn.BCEWithLogitsLoss()

    def extract_edges_on_gpu(self, masks):
        """
        利用 GPU 进行形态学梯度提取边缘：边缘 = 膨胀(Mask) - 原Mask
        masks shape: (Batch_Size, 1, H, W)，且值必须为 0 或 1
        """
        # 利用 3x3 的最大池化模拟形态学膨胀
        dilation = F.max_pool2d(masks, kernel_size=3, stride=1, padding=1)
        # 膨胀后的图像减去原图，剩下的就是一圈只有 1 个像素宽的外边缘
        edges = dilation - masks
        return edges

    def dice_loss(self, pred_logits, target_masks, smooth=1e-5):
        """
        Dice Loss：专门对抗遥感建筑物正负样本极度不平衡的神器
        """
        # 将 Logits 映射到 0-1 的概率区间
        pred_probs = torch.sigmoid(pred_logits)

        # 展平以便计算
        pred_flat = pred_probs.reshape(pred_probs.size(0), -1)
        target_flat = target_masks.reshape(target_masks.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2. * intersection + smooth) / (union + smooth)
        # 返回 1 - Dice 作为 Loss
        return 1.0 - dice.mean()

    def forward(self, pred_mask, pred_edge, target_mask):
        """
        pred_mask: 网络的 Mask 分支输出 (B, 1, H, W) -> Logits
        pred_edge: 网络的 Edge 分支输出 (B, 1, H, W) -> Logits
        target_mask: 数据集自带的真实标签 (B, 1, H, W) -> 0 或 1
        """
        # 2. 动态生成边缘的真实标签 (Ground Truth)
        # 预处理 target_mask：确保它是单通道的二值图 (B, 1, H, W)，以便正确提取单通道边缘
        if target_mask.dim() == 4 and target_mask.shape[1] > 1:
            # 如果是 One-Hot 格式 (B, C, H, W)，取最大值所在的通道索引，转为 (B, 1, H, W)
            target_mask_binary = target_mask.argmax(dim=1, keepdim=True).float()
        else:
            target_mask_binary = target_mask
        target_edge = self.extract_edges_on_gpu(target_mask_binary)

        # 3. 计算 Mask 损失 (BCE + Dice)
        mask_bce = self.bce_loss(pred_mask, target_mask)
        mask_dice = self.dice_loss(pred_mask, target_mask)
        loss_mask = mask_bce + mask_dice

        # 4. 计算 Edge 损失 (BCE)
        # 边缘像素极其稀疏，BCE 已经能产生很好的惩罚效果
        loss_edge = self.bce_loss(pred_edge, target_edge)

        # 5. 总损失计算
        # 公式: L_total = L_mask + lambda * L_edge
        total_loss = loss_mask + self.edge_weight * loss_edge

        return total_loss, loss_mask, loss_edge

def write_images_to_tensorboard(writer, epoch, image=None, seg_diffused=None, seg_predicted=None, seg_gt=None, datasplit='validation', dataset_name='cityscapes'):
        """Writes images to TensorBoard"""
        # decode segmap based on dataset
        if dataset_name == 'cityscapes':
            decode_segmap = decode_segmap_cityscapes
        elif dataset_name == 'uavid':
            decode_segmap = decode_segmap_uavid
        elif dataset_name == 'vaihingen':
            decode_segmap = decode_segmap_vaihingen
        elif dataset_name == 'inria':
            decode_segmap = decode_segmap_inria
        elif dataset_name == 'whu':
            decode_segmap = decode_segmap_inria
        elif dataset_name == 'massa':
            decode_segmap = decode_segmap_inria
        else:
            raise NotImplementedError('Dataset {} not implemented'.format(dataset_name))
        if image is not None:
            image = torchvision.utils.make_grid(image, normalize=True) # normalize to [0,1] and convert to uint8
            writer.add_images('{}/image'.format(datasplit), image, epoch, dataformats='CHW')
        if seg_diffused is not None:
            seg_diffused = decode_segmap(seg_diffused, is_one_hot=True)
            writer.add_images('{}/seg_diffused'.format(datasplit), seg_diffused, epoch, dataformats='CHW')
        if seg_predicted is not None:
            seg_predicted = decode_segmap(seg_predicted, is_one_hot=True)
            writer.add_images('{}/seg_predicted'.format(datasplit), seg_predicted, epoch, dataformats='CHW')
        if seg_gt is not None:
            seg_gt = decode_segmap(seg_gt, is_one_hot=False)
            writer.add_images('{}/seg_gt'.format(datasplit), seg_gt, epoch, dataformats='CHW')
def denoise_loop_scales(model, device, network_config, images):
    """Denoises all scales for a single timestep"""
    # Calculate scale sizes (smallest first)
    scale_sizes = [(images.shape[2] // (2**(network_config.n_scales - i -1)), images.shape[3] // (2**(network_config.n_scales - i -1))) for i in range(network_config.n_scales)]

    # Initialize first prediction (random noise)
    seg_previous_scaled = torch.rand(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

    # Initialize built in ensemble
    seg_denoised_ensemble = torch.zeros(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

    # Denoise whole segmentation map in steps
    for timestep in range(network_config.n_timesteps): # for each step
        
        for scale in range(network_config.n_scales): # for each scale
            # Resize to current scale
            images_scaled = F.interpolate(images, size=scale_sizes[scale], mode='bilinear', align_corners=False)
            seg_previous_scaled = F.interpolate(seg_previous_scaled.float(), size=scale_sizes[scale], mode='bilinear', align_corners=False).softmax(dim=1)

            # Diffuse
            t = torch.tensor([(network_config.n_timesteps - (timestep + scale/network_config.n_scales)) / network_config.n_timesteps]) # time step
            seg_diffused = diffuse(seg_previous_scaled, t)  
            # Denoise
            seg_denoised = denoise_scale(model, device, seg_diffused, images_scaled, t, patch_size=network_config.max_patch_size)

            # Update the previous segmentation map
            seg_previous_scaled = seg_denoised
        
        # Add to ensemble
        if network_config.built_in_ensemble:
            if timestep == 0:
                seg_denoised_ensemble = seg_denoised
            else:
                seg_denoised_ensemble = seg_denoised_ensemble / 2 + seg_denoised / 2
            
            seg_previous_scaled = seg_denoised_ensemble
    
    return seg_denoised

def denoise_linear_scales(model, device, network_config, images):
    """Denoises one scale at a each timestep"""
    # Calculate scale sizes (smallest first)
    scale_sizes = [(images.shape[2] // (2**(network_config.n_scales - i -1)), images.shape[3] // (2**(network_config.n_scales - i -1))) for i in range(network_config.n_scales)]

    # Initialize first prediction (random noise)
    seg_previous_scaled = torch.rand(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

    # Denoise whole segmentation map in steps
    for timestep in range(network_config.n_timesteps): # for each step
        # Get the current scale
        timesteps_per_scale = math.ceil(network_config.n_timesteps / network_config.n_scales)
        scale = timestep // timesteps_per_scale
        
        # Resize to current scale
        if timestep % timesteps_per_scale == 0:
            images_scaled = F.interpolate(images, size=scale_sizes[scale], mode='bilinear', align_corners=False)
            seg_previous_scaled = F.interpolate(seg_previous_scaled.float(), size=scale_sizes[scale], mode='bilinear', align_corners=False)

        # Diffuse
        t = torch.tensor([(network_config.n_timesteps - (timestep + scale/network_config.n_scales)) / network_config.n_timesteps]) # time step
        seg_diffused = diffuse(seg_previous_scaled, t)
        # Denoise
        seg_denoised = denoise_scale(model, device, seg_diffused, images_scaled, t, patch_size=network_config.max_patch_size)

        # Update the previous segmentation map
        seg_previous_scaled = seg_denoised

    return seg_denoised

def denoise(model, device, network_config, images):
        """Denoises the segmentation map"""
        if network_config.scale_procedure == 'loop':
            seg_denoised = denoise_loop_scales(model, device, network_config, images)
        elif network_config.scale_procedure == 'linear':
            seg_denoised = denoise_linear_scales(model, device, network_config, images)

        return seg_denoised

class Evaluator:
    """Evaluates the performance of a model"""
    def __init__(self, model, network_config, device, dataset_selection=None, test_data_loader=None, validation_data_loader=None, writer=None):
        self.model = model
        self.network_config = network_config
        self.device = device
        self.dataset_selection = dataset_selection
        self.test_data_loader = test_data_loader
        self.validation_data_loader = validation_data_loader
        self.writer = writer
    def evaluate(self, data_loader, epoch=1, is_test=True, ensemble=1): # epoch=None
        """Evaluates the model on the given dataset"""
        model = self.model
        network_config = self.network_config
        model.eval()

        if self.dataset_selection == 'cityscapes':
            ignore_index = 19
            n_ignore = 1
        else:
            ignore_index = None
            n_ignore = 0
            # 获取设备信息
        device = self.device  # 使用Evaluator的设备属性
        #这儿我把task改成了binary,,average='macro'
        jaccard_index = JaccardIndex(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore, ignore_index=ignore_index,average='macro').to(device)
        jaccard_per_class = JaccardIndex(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore, ignore_index=ignore_index, average='none').to(device)
        f1_score = F1Score(task="multiclass",num_classes=data_loader.dataset.n_classes + n_ignore, mdmc_average='samplewise',average='macro').to(device)
        accurate = Accuracy(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore,
                            mdmc_average='samplewise',average='micro').to(device)
        precision = Precision(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore,
                              mdmc_average='samplewise',average='macro').to(device)
        recall = Recall(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore,
                        mdmc_average='samplewise',average='macro').to(device)
        with torch.no_grad():
            pbar_eval = tqdm(enumerate(data_loader), total=len(data_loader), desc='{}'.format('Test' if is_test else 'Validation'), leave=is_test, bar_format='{l_bar}{bar:50}{r_bar}')
            for it, samples in pbar_eval:
                # Unpack the samples
                images, seg_gt = samples

                # 获取当前 batch 的图片路径
                batch_start_idx = it * data_loader.batch_size
                batch_image_paths = [data_loader.dataset.file_names[batch_start_idx + i]
                                     for i in range(len(images))]

                images = images.detach().cuda(non_blocking=True)
                seg_gt = seg_gt.detach().cuda(non_blocking=True)
                seg_denoised = model(images)
                # Compute loss
                seg_predicted = seg_denoised.view(seg_denoised.shape[0], seg_denoised.shape[1], -1).argmax(dim=1).to(device)
                seg_target = seg_gt.view(seg_gt.shape[0], -1).detach().cuda(non_blocking=True).to(device)

                jaccard_index.update(seg_predicted, seg_target)
                jaccard_per_class.update(seg_predicted, seg_target)
                f1_score.update(seg_predicted, seg_target)
                accurate.update(seg_predicted, seg_target)
                precision.update(seg_predicted, seg_target)
                recall.update(seg_predicted, seg_target)
                print('{} | {} | {} | {}| {} | {} | {}'.format(
                    "Test" if is_test else "Validation | Epoch: {}".format(epoch), jaccard_index.compute(),
                    jaccard_per_class.compute(), f1_score.compute(), accurate.compute(), precision.compute(),
                    recall.compute()))

                # 保存预测结果到 predict 文件夹
                predict_dir = os.path.join('predict', f'epoch_{epoch}')
                os.makedirs(predict_dir, exist_ok=True)

                for batch_idx in range(len(images)):
                    # 获取原始文件名（不含扩展名）
                    original_path = batch_image_paths[batch_idx]
                    original_filename = os.path.splitext(os.path.basename(original_path))[0]

                    # 从 seg_denoised (logits) 获取预测的类别掩码
                    # seg_denoised shape: (num_classes, H, W)
                    pred_logits = seg_denoised[batch_idx].cpu()
                    # 取概率最大的类别作为预测结果，得到 (H, W) 的二值图像
                    pred_mask = torch.argmax(pred_logits, dim=0).numpy()

                    # 保存为PNG格式的黑白二值图像，使用原始文件名 + _seg_denoised 后缀
                    save_path = os.path.join(predict_dir, f"{original_filename}_seg_denoised.png")
                    cv2.imwrite(save_path, (pred_mask * 255).astype(numpy.uint8))
                # Write images to tensorboard

                # if self.writer is not None:
                #     #if it < 8: 这儿屏蔽掉就可以存所以图片
                #     write_images_to_tensorboard(self.writer, epoch, image=images[0], seg_predicted=seg_denoised[0], seg_gt=seg_gt[0],dataset_name='inria', datasplit='validation/{}'.format(it))

        # Overall metrics
        jaccard_index_total = jaccard_index.compute()
        jaccard_per_class_total = jaccard_per_class.compute()
        f1_score_total = f1_score.compute()
        accurate_total = accurate.compute()
        precision_total = precision.compute()
        recall_total = recall.compute()
        print(f"JaccardIndex: {jaccard_index_total}, F1Score: {f1_score_total}, Accurate: {accurate_total}, Precision: {precision_total}, Recall: {recall_total}")
        # Text report
        report = 'Jaccard index: {:.4f} | F1 score: {:.4f}'.format(jaccard_index_total, f1_score_total)
        report_per_class = 'Jaccard index per class: {}'.format(jaccard_per_class_total)
        if self.writer is None:
            logging.log(logging.WARNING, report)
            logging.log(logging.WARNING, report_per_class)
        else:
            logging.info('{} | {} | {}'.format("Test" if is_test else "Validation | Epoch: {}".format(epoch), report, report_per_class))

        # Write to tensorboard
        if self.writer is not None:
            self.writer.add_scalar('{}/JaccardIndex'.format('test' if is_test else 'validation'), jaccard_index_total, epoch)
            self.writer.add_scalar('{}/F1Score'.format('test' if is_test else 'validation'), f1_score_total, epoch)
            self.writer.add_scalar('{}/Accurate'.format('test' if is_test else 'validation'), accurate_total, epoch)
            self.writer.add_scalar('{}/Precision'.format('test' if is_test else 'validation'), precision_total,
                                   epoch)
            self.writer.add_scalar('{}/Recall'.format('test' if is_test else 'validation'),recall_total, epoch)
    #之前的 evaluate
    # def evaluate(self, data_loader, epoch=1, is_test=True, ensemble=1): # epoch=None
    #     """Evaluates the model on the given dataset"""
    #     model = self.model
    #     network_config = self.network_config
    #     model.eval()
    #
    #     if self.dataset_selection == 'cityscapes':
    #         ignore_index = 19
    #         n_ignore = 1
    #     else:
    #         ignore_index = None
    #         n_ignore = 0
    #     #这儿我把task改成了binary,,average='macro'
    #     jaccard_index = JaccardIndex(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore, ignore_index=ignore_index,average='macro')
    #     jaccard_per_class = JaccardIndex(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore, ignore_index=ignore_index, average='none')
    #     f1_score = F1Score(task="multiclass",num_classes=data_loader.dataset.n_classes + n_ignore, mdmc_average='samplewise',average='macro')
    #     accurate = Accuracy(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore,
    #                         mdmc_average='samplewise',average='micro')
    #     precision = Precision(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore,
    #                           mdmc_average='samplewise',average='macro')
    #     recall = Recall(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore,
    #                     mdmc_average='samplewise',average='macro')
    #     with torch.no_grad():
    #         pbar_eval = tqdm(enumerate(data_loader), total=len(data_loader), desc='{}'.format('Test' if is_test else 'Validation'), leave=is_test, bar_format='{l_bar}{bar:50}{r_bar}')
    #         for it, samples in pbar_eval:
    #             # Unpack the samples
    #             images, seg_gt = samples
    #
    #             seg_denoised = denoise(model, self.device, network_config, images)
    #
    #             # Ensamble
    #             for i in range(ensemble-1):
    #                 seg_denoised += denoise(model, self.device, network_config, images)
    #             seg_denoised /= ensemble
    #             # Compute loss
    #             seg_predicted = seg_denoised.view(seg_denoised.shape[0], seg_denoised.shape[1], -1).argmax(dim=1)
    #             seg_target = seg_gt.view(seg_gt.shape[0], -1)
    #
    #             jaccard_index.update(seg_predicted, seg_target)
    #             jaccard_per_class.update(seg_predicted, seg_target)
    #             f1_score.update(seg_predicted, seg_target)
    #             accurate.update(seg_predicted, seg_target)
    #             precision.update(seg_predicted, seg_target)
    #             recall.update(seg_predicted, seg_target)
    #             print('{} | {} | {} | {}| {} | {} | {}'.format(
    #                 "Test" if is_test else "Validation | Epoch: {}".format(epoch), jaccard_index.compute(),
    #                 jaccard_per_class.compute(), f1_score.compute(), accurate.compute(), precision.compute(),
    #                 recall.compute()))
    #             # Write images to tensorboard
    #             if self.writer is not None:
    #                 #if it < 8: 这儿屏蔽掉就可以存所以图片
    #                 write_images_to_tensorboard(self.writer, epoch, image=images[0], seg_predicted=seg_denoised[0], seg_gt=seg_gt[0],dataset_name='inria', datasplit='validation/{}'.format(it))
    #
    #
    #     # Overall metrics
    #     jaccard_index_total = jaccard_index.compute()
    #     jaccard_per_class_total = jaccard_per_class.compute()
    #     f1_score_total = f1_score.compute()
    #     accurate_total = accurate.compute()
    #     precision_total = precision.compute()
    #     recall_total = recall.compute()
    #     print(f"JaccardIndex: {jaccard_index_total}, F1Score: {f1_score_total}, Accurate: {accurate_total}, Precision: {precision_total}, Recall: {recall_total}")
    #     # Text report
    #     report = 'Jaccard index: {:.4f} | F1 score: {:.4f}'.format(jaccard_index_total, f1_score_total)
    #     report_per_class = 'Jaccard index per class: {}'.format(jaccard_per_class_total)
    #     if self.writer is None:
    #         logging.log(logging.WARNING, report)
    #         logging.log(logging.WARNING, report_per_class)
    #     else:
    #         logging.info('{} | {} | {}'.format("Test" if is_test else "Validation | Epoch: {}".format(epoch), report, report_per_class))
    #
    #     # Write to tensorboard
    #     if self.writer is not None:
    #         self.writer.add_scalar('{}/JaccardIndex'.format('test' if is_test else 'validation'), jaccard_index_total, epoch)
    #         self.writer.add_scalar('{}/F1Score'.format('test' if is_test else 'validation'), f1_score_total, epoch)
    #         self.writer.add_scalar('{}/Accurate'.format('test' if is_test else 'validation'), accurate_total, epoch)
    #         self.writer.add_scalar('{}/Precision'.format('test' if is_test else 'validation'), precision_total,
    #                                epoch)
    #         self.writer.add_scalar('{}/Recall'.format('test' if is_test else 'validation'),recall_total, epoch)
        
    def validate(self, epoch):
        """Evaluates the model on the validation dataset"""
        self.evaluate(self.validation_data_loader, epoch, is_test=False)

    def test(self, ensemble=1):
        """Evaluates the model on the test dataset"""
        self.evaluate(self.test_data_loader, is_test=True, ensemble=ensemble)


