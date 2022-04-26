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


'''
import argparse
import copy
import csv
import os

import torch
from tqdm import tqdm
from torch import distributed
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
'''
import aimet_torch
import aimet_common
import numpy as np
from aimet_torch import bias_correction
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantParams, QuantizationSimModel 
from aimet_common.defs import QuantScheme
#from nets import nn
#from utils import util



def test(model,  eval_iterations: int):
    '''
    if model is None:
        model = torch.load('weights/best_pt.pt', map_location='cuda')['model'].float().eval()
    '''

    '''
    input_shape = (1, 3, 384, 384)
    dummy_input = torch.rand(input_shape).cuda()
    quantsim = QuantizationSimModel(model=model, quant_scheme='tf',
                                    dummy_input=dummy_input, rounding_mode='nearest',
                                    default_output_bw=8, default_param_bw=8)

    quantsim.compute_encodings(forward_pass_callback=partial(evaluator, use_cuda=use_cuda),
                               forward_pass_callback_args=iterations)
    

    quantsim.export(path=logdir, filename_prefix='resnet_encodings', dummy_input=dummy_input.cpu())
    accuracy = evaluator(quantsim.model, use_cuda=use_cuda)
    '''

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_transforms = transforms.Compose([transforms.Resize(416),
                                                       transforms.CenterCrop(384),
                                                       transforms.ToTensor(), 
                                                       normalize])
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                   transforms.Compose([transforms.Resize(416),
                                                       transforms.CenterCrop(384),
                                                       transforms.ToTensor(), normalize]))
    loader = data.DataLoader(dataset, 1, num_workers=8, pin_memory=True)
    top1 = util.AverageMeter()
    top5 = util.AverageMeter()

    from glob import glob
    from PIL import Image
    with torch.no_grad():
        import json
        with open("./Imagenet_label.json", 'r') as f:
            temp = json.load(f)
        class_index_names = {}
        for item in temp:
            if not item in class_index_names:
                class_index_names[temp[item][0]] = item

        count = 0 
        # import tqdm
        for images, target in tqdm(loader, ('%10s' * 2) % ('acc@1', 'acc@5')):
        # for filename in tqdm(sorted(glob(data_dir))):   
            
            if count == eval_iterations:
                break
            count = count + 1

            # target = filename.replace("./384x384/", "").split("/")[0]
            # print(class_index_names[target])
            # target = torch.tensor([int(class_index_names[target])])
            
            # img = Image.open(filename)
            # images = val_transforms(img)
            # images = images.unsqueeze(0)

            acc1, acc5 = batch(images, target, model)
            torch.cuda.synchronize()
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

        acc1, acc5 = top1.avg, top5.avg
        print('%10.3g' * 2 % (acc1, acc5))
    if model is None:
        torch.cuda.empty_cache()
    else:
        return acc1, acc5
        
        
def qat(args):
  model = torch.load('weights/best_pt.pt', map_location='cuda')['model'].float().eval()
  input_shape = (4, 3, 513, 513)
  
  # modelScript = torch.jit.trace(model.to(torch.device('cpu')), torch.tensor(np.random.rand(1, 3, 384, 384).astype(np.float32)))
  # print("#"*100)
  
  dummy_input = torch.rand(input_shape).cuda()
  equalize_model(model, input_shape)
  
  print("Quantsim Started")
  
  # quant_scheme = QuantScheme.post_training_tf_enhanced
  quantsim = QuantizationSimModel(model=model, quant_scheme="tf",
                                  dummy_input=dummy_input, rounding_mode='nearest',
                                  default_output_bw=8, default_param_bw=8)
                                  
  print("Quantsim Completed")
  
  print("Compute Encodings Started")
  
  #quantsim.compute_encodings(forward_pass_callback=partial(evaluator, use_cuda=use_cuda),forward_pass_callback_args=iterations)
  quantsim.compute_encodings(forward_pass_callback=test, forward_pass_callback_args=50000000000)
  
  print("Compute Encodings Completed")
  
  ptq_accuracy = test(quantsim.model, 10000000000)
  
  print("PTQ Accuracy",ptq_accuracy)
  
  print("QAT Started")
  train(args,quantsim.model)
  print("QAT Completed")
  
  '''
  trained_model = nn.EfficientNet(args)
  trained_model.load_state_dict(torch.load("/home/ava/sarala/EffcientNetV2/qat/AIMET_1.19.1/tf_cl_16_16/qat/torch_save/qat.pt", map_location='cuda'))
  accuracy = test(trained_model, 10)
  '''
  #trained_model = torch.load("/home/ava/sarala/EffcientNetV2/qat/AIMET_1.19.1/tf_cl_16_16/qat/torch_save/effnet_v2-s_qat.pt", map_location='cuda').float().eval()
  #trained_model = torch.load("/home/ava/sarala/EffcientNetV2/qat/AIMET_1.19.1/tf_cle_16_16/qat/effnet_v2-s_qat.pt", map_location='cuda').float().eval()
  trained_model = torch.load("/home/ava/sarala/EffcientNetV2/qat/AIMET_1.19.1/tf_cle_8_8/qat/effnet_v2-s_qat.pt", map_location='cuda').float().eval()
  
  
  qat_accuracy = test(trained_model, 10000000000)
  print("QAT Accuracy",qat_accuracy)
  '''
  quantsim.export(path="/home/ava/sarala/EffcientNetV2/qat/AIMET_1.19.1/tf_cl_16_16/ptq/",
   filename_prefix='effnet_v2-S', dummy_input=torch.rand(input_shape, device="cpu"),
                  onnx_export_args=(aimet_torch.onnx_utils.OnnxExportApiArgs (opset_version=11)))
  '''
  
  quantsim.export(path="/home/ava/sarala/EffcientNetV2/qat/AIMET_1.19.1/tf_cle_8_8/ptq/",
   filename_prefix='effnet_v2-S', dummy_input=torch.rand(input_shape, device="cpu"),
                  onnx_export_args=(aimet_torch.onnx_utils.OnnxExportApiArgs (opset_version=11)))
                  
                  
                  
  ''' 
  changed the line inside the torch library
       anaconda3\envs\efficientnetv2pytorch\Lib\site-packages\torch\nn\functional.py
       In Line number: 1742
        replace "return torch._C._nn.silu(input)" => "return input*torch.sigmoid(input)"
  '''

class Predictor(object):
    def __init__(self, args,ckpt ):
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
            print("===========",classes_weights_path)
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
            print("args.resume",args.resume)
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = ckpt
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
            print("===",image.size())
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


    def predict(self, epoch=1):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            self.evaluator.reset()

            image, target = sample['image'], sample['label']
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
            # Add batch sample into evaluator & count
            self.evaluator.add_batch(target, pred)
            # Acc = self.evaluator.Pixel_Accuracy()
            # Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            # FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            # composite name
            pic_output = str(mIoU) + '.' + str(np.random.randint(100))
            # raw picture
            img = np.transpose(image[0].detach().cpu().numpy(), (1, 2, 0))
            img *= (0.229, 0.224, 0.225)
            img += (0.485, 0.456, 0.406)
            img *= 255.0
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            mask = pred[0]
            # ground truth
            mask_gt = VOCSegmentation.fill_colormap(target[0])
            mask_gt = Image.fromarray(mask_gt)
            # predict mask
            mask = VOCSegmentation.fill_colormap(mask)
            mask = Image.fromarray(mask)

            print('label {} miou is {}'.format(i, mIoU))
            self._check_out_dir()
            file_path = 'output/{}/{}.jpg'
            img.save(file_path.format('raw', pic_output), format='jpeg')
            mask_gt.save(file_path.format('gt', pic_output), format='jpeg')
            mask.save(file_path.format('mask', pic_output), format='jpeg')

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
    False, help='disables CUDA training')
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
    #torch.manual_seed(args.seed)
    #tester = Predictor(args)
    #tester.validation()
    # print('Starting Epoch:', trainer.args.start_epoch)
    # print('Total Epoches:', trainer.args.epochs)
    # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
    #     trainer.training(epoch)
    #     if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
    #         trainer.validation(epoch)
    #tester.writer.close()
    
    ckpt = torch.load("/home/bmw/sarala/pytorch-deeplab-xception/weights/xception-b5690688.pth", map_location='cuda')
    input_shape = (4, 3, 513, 513)
    
    
    tester = Predictor(args,ckpt)
    tester.validation()
    
    # modelScript = torch.jit.trace(model.to(torch.device('cpu')), torch.tensor(np.random.rand(1, 3, 384, 384).astype(np.float32)))
    # print("#"*100)
    
    dummy_input = torch.rand(input_shape).cuda()
    equalize_model(model, input_shape)
    
    print("Quantsim Started")
    
    # quant_scheme = QuantScheme.post_training_tf_enhanced
    quantsim = QuantizationSimModel(model=model, quant_scheme="tf",
                                    dummy_input=dummy_input, rounding_mode='nearest',
                                    default_output_bw=8, default_param_bw=8)
                                    
    print("Quantsim Completed")
    
    '''
    print("Compute Encodings Started")
    
    float_tester = Predictor(args,model)
    #float_tester.validation()
    
    #quantsim.compute_encodings(forward_pass_callback=partial(evaluator, use_cuda=use_cuda),forward_pass_callback_args=iterations)
    quantsim.compute_encodings(forward_pass_callback=float_tester.validation)
    
    print("Compute Encodings Completed")
    
    
    float_tester = Predictor(args,model)
    #float_tester.validation()
    
    ptq_accuracy = tester.validation()(quantsim.model, 10000000000)
    
    print("PTQ Accuracy",ptq_accuracy)
     
    quantsim.export(path="/home/ava/sarala/EffcientNetV2/qat/AIMET_1.19.1/tf_cle_8_8/ptq/",
     filename_prefix='effnet_v2-S', dummy_input=torch.rand(input_shape, device="cpu"),
                    onnx_export_args=(aimet_torch.onnx_utils.OnnxExportApiArgs (opset_version=11)))
                    
    '''         



if __name__ == "__main__":
    main()
