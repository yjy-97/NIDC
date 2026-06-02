import os
import argparse
import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms

from my_dataset import MyDataSet
from model.cluster import NIDC_base_dim64
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate

import warnings
warnings.filterwarnings('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path = r"/home/yejiayu/Work/Project/data/OASIS/train1"
    # train_images_path = r"/home/yejiayu/Work/Project/data/ADVSNC_5/fold_5/train"
    # train_images_path = r"/home/yejiayu/Work/Project/data/ADVSMCI_5/fold_1/train"
    # train_images_path = r"/home/yejiayu/Work/Project/data/NCVSMCI_5/fold_1/train"
    # train_images_path = r"/home/yejiayu/Work/Project/data/train"

    val_images_path = r"/home/yejiayu/Work/Project/data/OASIS/testt1"
    # val_images_path = r"/home/yejiayu/Work/Project/data/ADVSNC_5/fold_5/test"
    # val_images_path = r"/home/yejiayu/Work/Project/data/ADVSMCI_5/fold_1/test"
    # val_images_path = r"/home/yejiayu/Work/Project/data/NCVSMCI_5/fold_1/test"
    # val_images_path = r"/home/yejiayu/Work/Project/data/test"

    # img_size = 224
    batch_size = args.batch_size

    # 实例化训练数据集
    train_dataset = MyDataSet(train_images_path)

    # 实例化验证数据集
    val_dataset = MyDataSet(val_images_path)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = NIDC_base_dim64().to(device)

    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)["model"]
    #     # 删除有关分类类别的权重
    #     for k in list(weights_dict.keys()):
    #         if "head" in k:
    #             del weights_dict[k]
    #     model.load_state_dict(weights_dict, strict=False)

    # if args.freeze_layers:
    # for name, para in model.named_parameters():
    # 除head外，其他权重全部冻结
    # if "head" not in name:
    # para.requires_grad_(False)
    # else:
    # print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.95, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=5)

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(model.state_dict(), "./weights/best_model_AH.pth")
            best_acc = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    # 0.005
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=5e-2)

    setup_seed(3407)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")

    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str,
                        default=r'/mnt/public/home/wangqx/Yejiayu/mri/ConvNeXt/convnext_base_1k_224_ema.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
