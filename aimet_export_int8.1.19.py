import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from PIL import Image
from dataloaders.datasets.pascal import VOCSegmentation

import os

import aimet_torch
import aimet_common
import numpy as np
from aimet_torch import bias_correction
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantParams, QuantizationSimModel 
from aimet_common.defs import QuantScheme

class Predictor(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.train_set_images, self.val_loader, self.val_set_images, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def _check_dir(self, dir_name):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    def _check_out_dir(self):
        self._check_dir('output')
        self._check_dir('output/raw')
        self._check_dir('output/gt')
        self._check_dir('output/mask')
       
    
    def validation(self, epoch=1):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            if(i==len(tbar)-1):
              break
            image, target = sample['image'], sample['label']
            #print("===",image.size())
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)



    
def validation(model,  args):
        tester = Predictor(args)
        model.eval()
        evaluator = Evaluator(21)
        evaluator.reset()
        tbar = tqdm(tester.val_loader, desc='\r')
        test_loss = 0.0
        test_loss_bar=0.0
        for i, sample in enumerate(tbar):
            if(i==len(tbar)-1):
              break
            image, target = sample['image'], sample['label']
            #print("===",image.size())
            if args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                #print("model",type(model))
                #print("model",model)
                output = model(image)
            criterion = SegmentationLosses().build_loss(mode='ce')
            loss = criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            test_loss_bar=(test_loss / (i + 1))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(target, pred)
        print("test_loss",test_loss)
        print("test_loss_bar",test_loss_bar)
        # Fast test during the training
        #Acc = evaluator.Pixel_Accuracy()
        #Acc_class = evaluator.Pixel_Accuracy_Class()
        #mIoU = evaluator.Mean_Intersection_over_Union()
        #FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        #self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        #self.writer.add_scalar('val/mIoU', mIoU, epoch)
        #self.writer.add_scalar('val/Acc', Acc, epoch)
        #self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        #self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        #print('Validation:')
        #batch_size=4
        #print('[Epoch: %d, numImages: %5d]' % (epoch, i * batch_size + image.data.shape[0]))
        #print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        #print('Loss: %.3f' % test_loss) 
        
 

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    True, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    print(args)
    print(type(args))
    torch.manual_seed(args.seed)
    tester = Predictor(args)
    #tester.validation()
    
    
    model = tester.model
    input_shape = (4, 3, 513, 513)
  
    dummy_input = torch.rand(input_shape)
    equalize_model(model, input_shape)
  
    print("Quantsim Started")
  
    # quant_scheme = QuantScheme.post_training_tf_enhanced
    quantsim = QuantizationSimModel(model=model, quant_scheme="tf",
                                  dummy_input=dummy_input, rounding_mode='nearest',
                                  default_output_bw=8, default_param_bw=8)
                                  
    print("Quantsim Completed")
  
    
    print("Compute Encodings Started")
  
    #quantsim.compute_encodings(forward_pass_callback=partial(evaluator, use_cuda=use_cuda),forward_pass_callback_args=iterations)
    quantsim.compute_encodings(forward_pass_callback=validation, forward_pass_callback_args=args)
  
    print("Compute Encodings Completed")
  
    print("PTQ Accuracy")
    validation(quantsim.model, args)
  
    #print("PTQ Accuracy",ptq_accuracy)
  
    
    quantsim.export(path="/home/bmw/sarala/pytorch-deeplab-xception/AIMET_1.19.1/ptq/tf_cle_8_8/",
     filename_prefix='deeplabv3_exception', dummy_input=torch.rand(input_shape, device="cpu"),
                  onnx_export_args=(aimet_torch.onnx_utils.OnnxExportApiArgs (opset_version=11)))
           
                  
    # print('Starting Epoch:', trainer.args.start_epoch)
    # print('Total Epoches:', trainer.args.epochs)
    # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
    #     trainer.training(epoch)
    #     if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
    #         trainer.validation(epoch)
    tester.writer.close()


if __name__ == "__main__":
    main()
