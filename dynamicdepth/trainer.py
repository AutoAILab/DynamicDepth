import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import pdb
import numpy as np
import time
import random
from tqdm import tqdm
import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from manydepth.rigid_warp import forward_warp
import scipy

import json

from .utils import readlines, sec_to_hm_str
from .layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss, compute_depth_errors

from manydepth import datasets, networks
import matplotlib.pyplot as plt


_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer:
    def __init__(self, options):
        self.opt = options
        wandb.init(project='InstaMD')
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        self.train_teacher_and_pose = not self.opt.freeze_teacher_and_pose
        if self.train_teacher_and_pose:
            print('using adaptive depth binning!')
            self.min_depth_tracker = 0.1
            self.max_depth_tracker = 10.0
        else:
            print('fixing pose network and monocular network!')

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = [0]
        if self.opt.use_future_frame == 'true':
            self.matching_ids.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)
            if idx not in frames_to_load:
                frames_to_load.append(idx)

        print('Loading frames: {}'.format(frames_to_load))

        # MODEL SETUP
        self.models["encoder"] = networks.ResnetEncoderMatching(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            input_height=self.opt.height, input_width=self.opt.width,
            adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
            depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins)
        self.models["encoder"].to(self.device)

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)

        if not self.opt.train_teacher_only:
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["mono_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained")
        self.models["mono_encoder"].to(self.device)

        self.models["mono_depth"] = \
            networks.DepthDecoder(self.models["mono_encoder"].num_ch_enc, self.opt.scales)
        self.models["mono_depth"].to(self.device)

        if self.train_teacher_and_pose:
            self.parameters_to_train += list(self.models["mono_encoder"].parameters())
            self.parameters_to_train += list(self.models["mono_depth"].parameters())

        self.models["pose_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained",
                                   num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)

        self.models["pose"] = \
            networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                 num_input_features=1,
                                 num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)

        if self.train_teacher_and_pose:
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mono_weights_folder is not None:
            self.load_mono_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # DATA
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                         "cityscapes_InstaDM": datasets.CityscapesInstaDMDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        if self.opt.split in ['cityscapes_preprocessed', 'cityscapes_instadm']:
            fpath = os.path.join("splits", 'cityscapes', "{}_files.txt")
            train_filenames = readlines(fpath.format("train"))
        else:
            fpath = os.path.join("splits", self.opt.split, "{}_files.txt")
            train_filenames = readlines(fpath.format("withmask"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=not self.opt.export, img_ext=img_ext, feat_warp=self.opt.feat_warp and self.opt.split=='cityscapes_preprocessed')
        if self.opt.export:
            self.opt.batch_size = 1
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=False,#not self.opt.export,
            num_workers=self.opt.num_workers, pin_memory=False, drop_last=True,
            worker_init_fn=seed_worker)
        if self.opt.split == 'cityscapes_instadm':
            val_filenames = readlines('splits/cityscapes/test_files.txt')
            val_dataset = datasets.CityscapesInstaDMEvalDataset(self.opt.eval_data_path,
                                                     val_filenames,
                                                     self.opt.height, self.opt.width,
                                                     [0, -1], 4,
                                                     is_train=False) #, mask_noise=self.opt.mask_noise=='doj')
        elif self.opt.split == 'cityscapes_preprocessed':
            val_filenames = readlines('splits/cityscapes/test_files.txt')
            val_dataset = datasets.CityscapesEvalDataset(self.opt.eval_data_path, val_filenames,
                                                     self.opt.height, self.opt.width,
                                                     [0, -1], 4,
                                                     is_train=False,
                                                     feat_warp=self.opt.feat_warp)
        else:
            val_filenames = readlines(os.path.join('splits', self.opt.eval_split, "test_files.txt"))
            val_dataset = self.dataset(
                self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                frames_to_load, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, 1, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=False, drop_last=False)
        
        self.train_filenames = train_filenames
        self.val_filenames = val_filenames

        if self.opt.split in ['cityscapes_preprocessed', 'cityscapes_instadm']:
            print('loading cityscapes gt depths individually due to their combined size!')
            self.gt_depths = 'splits/cityscapes/gt_depths/'
        else:
            gt_path = os.path.join('splits', self.opt.eval_split, "gt_depths.npz")
            self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim == 'true':
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()
        wandb.config.update(self.opt)
        self.best = 10.0
        self.doj_best = 10.0
        torch.autograd.set_detect_anomaly(True)

    def set_train(self):
        """Convert all models to training mode
        """
        for k, m in self.models.items():
            if self.train_teacher_and_pose:
                m.train()
            else:
                # if teacher + pose is frozen, then only use training batch norm stats for
                # multi components
                if k in ['depth', 'encoder']:
                    m.train()

    def freeze_teacher(self):
        if self.train_teacher_and_pose:
            assert not self.opt.train_teacher_only
            self.train_teacher_and_pose = False
            print('freezing teacher and pose networks!')

            # here we reinitialise our optimizer to ensure there are no updates to the
            # teacher and pose networks
            self.parameters_to_train = []
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, self.opt.scheduler_step_size, 0.1)

            # set eval so that teacher + pose batch norm is running average
            self.set_eval()
            # set train so that multi batch norm is in train mode
            self.set_train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        if self.opt.load_weights_folder is not None:
            self.val()
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            if self.epoch == self.opt.freeze_teacher_epoch:
                self.freeze_teacher()

            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs, is_train=True)
            if not self.opt.export:
                self.model_optimizer.zero_grad()
                losses["loss"].backward()
                self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 4000
            late_phase = self.step % 2000 == 0

            if (early_phase or late_phase) and self.step > 0 and not self.opt.export:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()
                if self.opt.save_intermediate_models and late_phase:
                    self.save_model(save_step=True)
                if self.step == self.opt.freeze_teacher_step:
                    self.freeze_teacher()
            
            if self.opt.export and self.step==0:
                self.val()
                    
            self.step += 1
        self.model_lr_scheduler.step()

    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}

        # predict poses for all frames
        if self.train_teacher_and_pose:
            pose_pred = self.predict_poses(inputs, None, is_train or self.opt.export)
        else:
            with torch.no_grad():
                pose_pred = self.predict_poses(inputs, None, is_train or self.opt.export)
        outputs.update(pose_pred)
        mono_outputs.update(pose_pred)

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

        # apply static frame and zero cost volume augmentation
        batch_size = len(lookup_frames)
        augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()
        if is_train and not self.opt.no_matching_augmentation == 'true':
            for batch_idx in range(batch_size):
                rand_num = random.random()
                # static camera augmentation -> overwrite lookup frames with current frame
                if rand_num < 0.25:
                    replace_frames = \
                        [inputs[('color', 0, 0)][batch_idx] for _ in self.matching_ids[1:]]
                    replace_frames = torch.stack(replace_frames, 0)
                    lookup_frames[batch_idx] = replace_frames
                    augmentation_mask[batch_idx] += 1
                # missing cost volume augmentation -> set all poses to 0, the cost volume will
                # skip these frames
                elif rand_num < 0.5:
                    relative_poses[batch_idx] *= 0
                    augmentation_mask[batch_idx] += 1
        outputs['augmentation_mask'] = augmentation_mask

        min_depth_bin = self.min_depth_tracker
        max_depth_bin = self.max_depth_tracker

        # single frame path
        if self.train_teacher_and_pose:
            feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
            mono_outputs.update(self.models['mono_depth'](feats))
        else:
            with torch.no_grad():
                feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
                mono_outputs.update(self.models['mono_depth'](feats))


        ############# warpping image based on teacher model predicted pose and depth ###############
        if (not self.opt.no_warp) and self.opt.dataset in ['cityscapes_preprocessed', 'kitti']:
            with torch.no_grad():
                _, teacher_depth = disp_to_depth(mono_outputs["disp", 0].detach().clone(), self.opt.min_depth, self.opt.max_depth)  # [12, 1, 192, 512]
                teacher_depth =teacher_depth.detach().clone()
                tgt_imgs = inputs["color", 0, 0].detach().clone()  # [12, 3, 192, 512]
                m1_pose = outputs[("cam_T_cam", 0, -1)][:, :3, :].detach().clone()  # [12, 3, 4]
                intrins = inputs[('K', 0)][:,:3,:3]  # [12, 3, 3]
                doj_mask = inputs["doj_mask"]  # [12, 1, 192, 512]
                tgt_imgs[doj_mask.repeat(1,3,1,1)==0] = 0
                img_w_m1, _, _ = forward_warp(tgt_imgs, teacher_depth, m1_pose, intrins, upscale=3, rotation_mode='euler', padding_mode='zeros')
                doj_maskm1 = inputs["doj_mask-1"].repeat(1,3,1,1)  # [12, 3, 192, 512]
                if self.opt.no_teacher_warp:
                    inputs['ori_color', -1, 0] = inputs["color", -1, 0].detach().clone()
                inputs["color", -1, 0][doj_maskm1==1] = 0
                if not self.opt.no_reproj_doj:
                    inputs["color", -1, 0][img_w_m1>0] = img_w_m1[img_w_m1>0]
                else:
                    inputs["color", -1, 0][img_w_m1>0] = 0
                inputs["color", -1, 0] = inputs["color", -1, 0].detach().clone()

                non_cv_aug = [augmentation_mask[:,0,0,0]==0][0]  # [12]
                if non_cv_aug.sum() > 0:
                    tgt_imgs_aug = inputs["color_aug", 0, 0].detach().clone()  # [12, 3, 192, 512]
                    tgt_imgs_aug[doj_mask.repeat(1,3,1,1)==0] = 0
                    imgaug_w_m1, _, _ = forward_warp(tgt_imgs_aug[non_cv_aug], teacher_depth[non_cv_aug], m1_pose[non_cv_aug], intrins[non_cv_aug], upscale=3, rotation_mode='euler', padding_mode='zeros')
                    warp_frame = lookup_frames[non_cv_aug][:,0,:,:,:].detach().clone()
                    warp_frame[doj_maskm1[non_cv_aug]==1] = 0
                    warp_frame[imgaug_w_m1>0] = imgaug_w_m1[imgaug_w_m1>0]
                    lookup_frames[non_cv_aug] = warp_frame.unsqueeze(1).detach().clone()

                if is_train:
                    p1_pose = outputs[("cam_T_cam", 0, 1)][:, :3, :].detach().clone()  # [12, 3, 4]
                    img_w_p1, _, _ = forward_warp(tgt_imgs, teacher_depth, p1_pose, intrins, upscale=3, rotation_mode='euler', padding_mode='zeros')
                    doj_maskp1 = inputs["doj_mask+1"].repeat(1,3,1,1)  # [12, 3, 192, 512]
                    if self.opt.no_teacher_warp:
                        inputs['ori_color', 1, 0] = inputs["color", -1, 0].detach().clone()
                    inputs["color", 1, 0][doj_maskp1==1] = 0
                    if not self.opt.no_reproj_doj:
                        inputs["color", 1, 0][img_w_p1>0] = img_w_p1[img_w_p1>0]
                    else:
                        inputs["color", 1, 0][img_w_p1>0] = 0
                    inputs["color", 1, 0] = inputs["color", 1, 0].detach().clone()
                #pdb.set_trace()
                #print('warp')
        ####################################################################################################
        

        if is_train:
            self.generate_images_pred(inputs, mono_outputs)
            mono_losses = self.compute_losses(inputs, mono_outputs, is_multi=False)
        else:
            _, mono_outputs["depth", 0, 0] = disp_to_depth(mono_outputs["disp", 0], self.opt.min_depth, self.opt.max_depth)

        # update multi frame outputs dictionary with single frame outputs
        for key in list(mono_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'mono_' + key[0]
                _key = tuple(_key)
                outputs[_key] = mono_outputs[key]

        # multi frame path
        if not self.opt.feat_warp:
            inputs["warp"] = None
        #lookup_frames = lookup_frames.permute([0,1,3,4,2])
        #lookup_frames[lookup_frames.sum(4) < 0.18]=0
        #lookup_frames = lookup_frames.permute([0,1,4,2,3])

        features, lowest_cost, confidence_mask = self.models["encoder"](inputs["color_aug", 0, 0],
                                                                        lookup_frames,
                                                                        relative_poses,
                                                                        inputs[('K', 2)],
                                                                        inputs[('inv_K', 2)],
                                                                        min_depth_bin=min_depth_bin,
                                                                        max_depth_bin=max_depth_bin,
                                                                        teacher_depth=outputs["mono_depth", 0, 0],
                                                                        mask_noise=self.opt.mask_noise,
                                                                        doj_mask=inputs["doj_mask"],
                                                                        cv_min=self.opt.cv_min=='true',
                                                                        aug_mask=augmentation_mask,
                                                                        set_1=self.opt.cv_set_1,
                                                                        pool=self.opt.cv_pool,
                                                                        pool_r=self.opt.cv_pool_radius,
                                                                        pool_th=self.opt.cv_pool_th,
                                                                        feat_warp=self.opt.feat_warp,
                                                                        warp=inputs["warp"])
        outputs.update(self.models["depth"](features))

        ############# warpping image based on multi frame model pose and depth ###############
        if self.opt.export:
            with torch.no_grad():
                _, multi_depth = disp_to_depth(outputs["disp", 0].detach(), self.opt.min_depth, self.opt.max_depth)  # [12, 1, 192, 512]
                multi_depth =multi_depth.detach()

                if not is_train:
                    save_path = 'visualization/pred/{}.npy'.format(int(inputs['index']))
                    np.save(save_path, multi_depth.squeeze().cpu().numpy())
           
        ####################################################################################################

        outputs["feat"] = features[-4]
        outputs["lowest_cost"] = F.interpolate(lowest_cost.unsqueeze(1),
                                               [self.opt.height, self.opt.width],
                                               mode="nearest")[:, 0]
        outputs["consistency_mask"] = F.interpolate(confidence_mask.unsqueeze(1),
                                                    [self.opt.height, self.opt.width],
                                                    mode="nearest")[:, 0]

        if is_train and not self.opt.export:
            if not self.opt.disable_motion_masking:
                outputs["consistency_mask"] = (outputs["consistency_mask"] *
                                            self.compute_matching_mask(outputs))

            self.generate_images_pred(inputs, outputs, is_multi=True)
            losses = self.compute_losses(inputs, outputs, is_multi=True)

            # update losses with single frame losses
            #print('mono reproj loss: ', mono_losses['reproj_loss/0'])
            #print('multi reproj loss: ', losses['reproj_loss/0'])
            #print('multi consis loss: ', losses['consistency_loss/0'])
            #print('')
            if self.train_teacher_and_pose:
                for key, val in mono_losses.items():
                    if not self.opt.no_multi_loss:
                        losses[key] += val
                    else:
                        losses[key] = val

            # update adaptive depth bins
            if self.train_teacher_and_pose:
                self.update_adaptive_depth_bins(outputs)
        else:
            losses = {}

        return outputs, losses

    def update_adaptive_depth_bins(self, outputs):
        """Update the current estimates of min/max depth using exponental weighted average"""

        min_depth = outputs[('mono_depth', 0, 0)].detach().min(-1)[0].min(-1)[0]
        max_depth = outputs[('mono_depth', 0, 0)].detach().max(-1)[0].max(-1)[0]

        min_depth = min_depth.mean().cpu().item()
        max_depth = max_depth.mean().cpu().item()

        # increase range slightly
        min_depth = max(self.opt.min_depth, min_depth * 0.9)
        max_depth = max_depth * 1.1

        self.max_depth_tracker = self.max_depth_tracker * 0.99 + max_depth * 0.01
        self.min_depth_tracker = self.min_depth_tracker * 0.99 + min_depth * 0.01

    def predict_poses(self, inputs, features=None, is_train=True):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if is_train:
            frameIDs = self.opt.frame_ids
        else:
            frameIDs = [0, -1]
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in frameIDs}
            for f_i in frameIDs[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        #mask = ((inputs['doj_mask']+inputs['doj_mask-1'])==0).float()
                        #pose_inputs = [pose_feats[f_i]*mask, pose_feats[0]*mask]
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        #mask = ((inputs['doj_mask']+inputs['doj_mask+1'])==0).float()
                        #pose_inputs = [pose_feats[0]*mask, pose_feats[f_i]*mask]
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

            # now we need poses for matching - compute without gradients
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.matching_ids}
            with torch.no_grad():
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in self.matching_ids[1:]:
                    if fi < 0:
                        #mask = ((inputs['doj_mask']+inputs['doj_mask-1'])==0).float()
                        #pose_inputs = [pose_feats[fi]*mask, pose_feats[fi + 1]*mask]
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                    else:
                        #mask = ((inputs['doj_mask']+inputs['doj_mask+1'])==0).float()
                        #pose_inputs = [pose_feats[fi - 1]*mask, pose_feats[fi]*mask]
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                    # set missing images to 0 pose
                    for batch_idx, feat in enumerate(pose_feats[fi]):
                        if feat.sum() == 0:
                            pose[batch_idx] *= 0

                    inputs[('relative_pose', fi)] = pose
        else:
            raise NotImplementedError

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        losses = {}
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = 0.0
            losses['doj/'+metric] = 0.0
        losses['doj/count'] = 0.0
        losses['dojpxs'] = 0
        losses['allpxs'] = 0
        total_batches = 0.0
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.val_loader)):
                total_batches += 1.0
                outputs, _ = self.process_batch(inputs)

                self.compute_depth_losses(inputs, outputs, losses, batch_idx, accumulate=True)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] /= total_batches
            losses['doj/'+metric] /= losses['doj/count']
            print(metric, ': ', losses[metric])
            print('doj/'+metric, ': ', losses['doj/'+metric])
            ###### done until here
        self.log('val', inputs, outputs, losses)
        if losses["de/abs_rel"] < self.best:
            self.best = losses["de/abs_rel"]
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!saving best result: ", self.best)
            for l, v in losses.items():
                wandb.log({"best{}".format(l): v}, step=self.step)
                writer = self.writers['val']
                writer.add_scalar("best{}".format(l), v, self.step)
                print(l, v)
            self.save_model("best")
            absrel = round(losses["de/abs_rel"] * 1000)
            if absrel < 120:
                self.save_model('absrel{}'.format(absrel))
        if losses["doj/de/abs_rel"] < self.doj_best:
            self.doj_best = losses["doj/de/abs_rel"]
            wandb.log({"bestdoj/de/absrel": self.doj_best}, step=self.step)
            print("***********************************dynamic objects depth best result: ", self.doj_best)
            print('doj pxs', losses['dojpxs'])
            print('all pxs', losses['allpxs'])
            doj = round(losses["doj/de/abs_rel"] * 1000)
            if doj < 140:
                self.save_model('doj{}'.format(doj))
        del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]
                if is_multi:
                    # don't update posenet based on multi frame prediction
                    T = T.detach()

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                if (not is_multi) and self.opt.no_teacher_warp and (not self.opt.train_teacher_only):
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("ori_color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True)
                else:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True)
                #pdb.set_trace()

                if not self.opt.disable_automasking:
                    if (not is_multi) and self.opt.no_teacher_warp and (not self.opt.train_teacher_only):
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("ori_color", frame_id, source_scale)]
                    else:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        if self.opt.zero_img:
            mask = (pred.sum(1) < 0.1).unsqueeze(1).repeat([1,3,1,1]).detach()
            pred = pred.clone()
            pred[mask] = 0
            target[mask] = 0
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim == 'true':
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    def compute_matching_mask(self, outputs):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        mono_output = outputs[('mono_depth', 0, 0)]
        matching_depth = 1 / outputs['lowest_cost'].unsqueeze(1).to(self.device)

        # mask where they differ by a large amount
        mask = ((matching_depth - mono_output) / mono_output) < 1.0
        mask *= ((mono_output - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]

    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking: # for stationary objects or static camera or low texture area
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    if (not is_multi) and self.opt.no_teacher_warp and (not self.opt.train_teacher_only):
                        pred = inputs[("ori_color", frame_id, source_scale)]
                    else:
                        pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # differently to Monodepth2, compute mins as we go
                    identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                              keepdim=True)
            else:
                identity_reprojection_loss = None

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                # differently to Monodepth2, compute mins as we go
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
            
            if self.opt.selec_reproj:
                maskm1 = (outputs[("color", -1, scale)].sum(1) < 0.1).detach()
                maskp1 = (outputs[("color", 1, scale)].sum(1) < 0.1).detach()
                maskand = (maskm1 * maskp1).detach()
                reprojection_loss[maskm1.unsqueeze(1)] = (reprojection_losses[:,1,:,:])[maskm1]
                reprojection_loss[maskp1.unsqueeze(1)] = (reprojection_losses[:,0,:,:])[maskp1]
                reprojection_loss[maskand.unsqueeze(1)] = 0
                if is_multi:
                    pdb.set_trace() #vis aachen 52 18


            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

            # find minimum losses from [reprojection, identity]
            reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                             identity_reprojection_loss)

            # find which pixels to apply reprojection loss to, and which pixels to apply
            # consistency loss to
            if is_multi:
                reprojection_loss_mask = torch.ones_like(reprojection_loss_mask)
                if not self.opt.disable_motion_masking:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                              outputs['consistency_mask'].unsqueeze(1))
                if not self.opt.no_matching_augmentation == 'true':
                    reprojection_loss_mask = (reprojection_loss_mask *
                                              (1 - outputs['augmentation_mask']))
                consistency_mask = (1 - reprojection_loss_mask).float()

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            # consistency loss:
            # encourage multi frame prediction to be like singe frame where masking is happening
            if is_multi:
                multi_depth = outputs[("depth", 0, scale)]
                # no gradients for mono prediction!
                mono_depth = outputs[("mono_depth", 0, scale)].detach()
                consistency_loss = torch.abs(multi_depth - mono_depth) * consistency_mask
                consistency_loss = consistency_loss.mean()

                # save for logging to tensorboard
                consistency_target = (mono_depth.detach() * consistency_mask +
                                      multi_depth.detach() * (1 - consistency_mask))
                consistency_target = 1 / consistency_target
                outputs["consistency_target/{}".format(scale)] = consistency_target
                losses['consistency_loss/{}'.format(scale)] = consistency_loss
            else:
                consistency_loss = 0

            losses['reproj_loss/{}'.format(scale)] = reprojection_loss

            loss += reprojection_loss + consistency_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        if self.opt.feat_loss=='true' and is_multi:
            losses["loss/feat_loss"] = self.get_feature_metric_loss(outputs["feat"], inputs[("color", 0, 0)])
            total_loss += losses["loss/feat_loss"]
        losses["loss"] = total_loss

        return losses

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy

    def get_feature_metric_loss(self, feature, img):
        b, _, h, w = feature.size()
        img = F.interpolate(img, (h, w), mode='area')

        feature_dx, feature_dy = self.gradient(feature)
        img_dx, img_dy = self.gradient(img)

        feature_dxx, feature_dxy = self.gradient(feature_dx)
        feature_dyx, feature_dyy = self.gradient(feature_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(feature_dx.abs() * torch.exp(-img_dx.abs().mean(1, True))) + \
                  torch.mean(feature_dy.abs() * torch.exp(-img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(feature_dxx.abs() * torch.exp(-img_dxx.abs().mean(1, True))) + \
                  torch.mean(feature_dxy.abs() * torch.exp(-img_dxy.abs().mean(1, True))) + \
                  torch.mean(feature_dyx.abs() * torch.exp(-img_dyx.abs().mean(1, True))) + \
                  torch.mean(feature_dyy.abs() * torch.exp(-img_dyy.abs().mean(1, True)))

        return -self.opt.feat_dis * smooth1+ self.opt.feat_cvt * smooth2

    def compute_depth_losses(self, inputs, outputs, losses, idx, accumulate=False):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        if not self.opt.train_teacher_only:
            _, depth_pred = disp_to_depth(outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
        else:
            _, depth_pred = disp_to_depth(outputs[("mono_disp", 0)], self.opt.min_depth, self.opt.max_depth)

        if self.opt.split in ['cityscapes_preprocessed', 'cityscapes_instadm']:
            gt_depth = np.load(os.path.join(self.gt_depths, str(idx).zfill(3) + '_depth.npy'))
            gt_height, gt_width = gt_depth.shape[:2]
            assert gt_height == 1024
            assert gt_width == 2048
            # crop ground truth to remove ego car -> this has happened in the dataloader for input
            # images
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]
        else:
            gt_depth = self.gt_depths[idx]
            gt_height, gt_width = gt_depth.shape[:2]

        depth_pred = torch.clamp(F.interpolate(depth_pred, [gt_height, gt_width], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach().squeeze()
        
        if self.opt.split in ['cityscapes_preprocessed', 'cityscapes_instadm']:
            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here
            gt_depth = gt_depth[256:, 192:1856]
            depth_pred = depth_pred[256:, 192:1856]

        if self.opt.eval_split in ["eigen", "eigen_benchmark"]:
            mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        elif self.opt.eval_split == 'cityscapes':
            mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
        
        gt_depth = torch.from_numpy(gt_depth).cuda()
        mask = torch.from_numpy(mask).cuda()

        depth_pred *= torch.median(gt_depth[mask]) / torch.median(depth_pred[mask])
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        inputs['doj_mask'] = F.interpolate(inputs['doj_mask'], [gt_height, gt_width])
        inputs['doj_mask'] = inputs['doj_mask'][0][0]
        if self.opt.split in ['cityscapes_preprocessed', 'cityscapes_instadm']:
            inputs['doj_mask'] = inputs['doj_mask'][256:,192:1856]
        doj_mask = mask * inputs['doj_mask'].bool()
        losses['dojpxs'] += doj_mask.sum()
        losses['allpxs'] += mask.sum()

        depth_errors = compute_depth_errors(gt_depth[mask], depth_pred[mask])
        doj_errors = compute_depth_errors(gt_depth[doj_mask], depth_pred[doj_mask])
        if doj_mask.sum() > 0:
            losses['doj/count'] += 1

        for i, metric in enumerate(self.depth_metric_names):
            if accumulate:
                losses[metric] += np.array(depth_errors[i].cpu())
                if doj_mask.sum() > 0:
                    losses['doj/'+metric] += np.array(doj_errors[i].cpu())
            else:
                losses[metric] = np.array(depth_errors[i].cpu())
        

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        if mode == 'val':
            frameIDs = [0, -1]
            BS = 1
        else:
            frameIDs = self.opt.frame_ids
            BS = self.opt.batch_size

        writer = self.writers[mode]
        for l, v in losses.items():
            wandb.log({"{}_{}".format(mode, l): v}, step=self.step)
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, BS)):  # write a maxmimum of four images
            s = 0  # log only max scale
            for frame_id in frameIDs:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, self.step)
                logimg = wandb.Image(inputs[("color", frame_id, s)][j].data.permute(1,2,0).cpu().numpy())
                wandb.log({"{}/color_{}_{}/{}".format(mode, frame_id, s, j): logimg}, step=self.step)
                if s == 0 and frame_id != 0 and mode == 'train':
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, self.step)
                    logimg = wandb.Image(outputs[("color", frame_id, s)][j].data.permute(1,2,0).cpu().numpy())
                    wandb.log({"{}/color_pred_{}_{}/{}".format(mode, frame_id, s, j): logimg}, step=self.step)

            disp = colormap(outputs[("disp", s)][j, 0])
            writer.add_image(
                "disp_multi_{}/{}".format(s, j),
                disp, self.step)
            logimg = wandb.Image(disp.transpose(1,2,0))
            wandb.log({"{}/disp_multi_{}/{}".format(mode, s, j): logimg}, step=self.step)
            disp = colormap(outputs[('mono_disp', s)][j, 0])
            writer.add_image(
                "disp_mono/{}".format(j),
                disp, self.step)
            logimg = wandb.Image(disp.transpose(1,2,0))
            wandb.log({"{}/disp_mono/{}".format(mode, j): logimg}, step=self.step)

            if outputs.get("lowest_cost") is not None:
                lowest_cost = outputs["lowest_cost"][j]

                consistency_mask = \
                    outputs['consistency_mask'][j].cpu().detach().unsqueeze(0).numpy()

                min_val = np.percentile(lowest_cost.numpy(), 10)
                max_val = np.percentile(lowest_cost.numpy(), 90)
                lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
                lowest_cost = colormap(lowest_cost)

                writer.add_image(
                    "lowest_cost/{}".format(j),
                    lowest_cost, self.step)
                writer.add_image(
                    "lowest_cost_masked/{}".format(j),
                    lowest_cost * consistency_mask, self.step)
                writer.add_image(
                    "consistency_mask/{}".format(j),
                    consistency_mask, self.step)
                logimg = wandb.Image(lowest_cost.transpose(1,2,0))
                wandb.log({"{}/lowest_cost/{}".format(mode, j): logimg}, step=self.step)
                logimg = wandb.Image((lowest_cost * consistency_mask).transpose(1,2,0))
                wandb.log({"{}/lowest_cost_masked/{}".format(mode, j): logimg}, step=self.step)
                logimg = wandb.Image(consistency_mask.transpose(1,2,0))
                wandb.log({"{}/consistency_mask/{}".format(mode, j): logimg}, step=self.step)
                if mode == 'train':
                    consistency_target = colormap(outputs["consistency_target/0"][j])
                    writer.add_image(
                        "consistency_target/{}".format(j),
                        consistency_target, self.step)
                    logimg = wandb.Image(consistency_target[0].transpose(1,2,0))
                    wandb.log({"{}/consistency_target/{}".format(mode, j): logimg}, step=self.step)
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, name=None, save_step=False):
        """Save model weights to disk
        """
        if name is not None:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(name))
        else:
            if save_step:
                save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch,
                                                                                        self.step))
            else:
                save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                # save estimates of depth bins
                to_save['min_depth_bin'] = self.min_depth_tracker
                to_save['max_depth_bin'] = self.max_depth_tracker
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_mono_model(self):

        model_list = ['pose_encoder', 'pose', 'mono_encoder', 'mono_depth']
        for n in model_list:
            print('loading {}'.format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            if n == 'encoder':
                min_depth_bin = pretrained_dict.get('min_depth_bin')
                max_depth_bin = pretrained_dict.get('max_depth_bin')
                print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print('setting depth bins!')
                    self.models['encoder'].compute_depth_bins(min_depth_bin, max_depth_bin)

                    self.min_depth_tracker = min_depth_bin
                    self.max_depth_tracker = max_depth_bin

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
