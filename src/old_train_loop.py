# import sys

# sys.path.append("E:\BRATS_Research\BraTS-2021\src\model")
# from model.unet import NvNet
# from loss import SoftDiceLossSquared

# import torch
# from torch.cuda.amp import autocast, GradScaler

# from tqdm import tqdm

# BATCH_SIZE = 4


# class AverageMeter(object):
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.sum = 0
#         self.avg = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def train_fn(loader, model, optimizer, criterion, scheduler, device):

# def testing_train_loop( model,optimizer, loader, criterion, scheduler,device):
#     losses = AverageMeter()
#     scaler = GradScaler()
#     model.to(device)
#     model.train()

#     loader = tqdm(loader,total = len(loader))

#     for step, batch in enumerate(loader):
#         image = batch['image'].to(device).float()
#         seg_truth = batch['mask'].to(device)

#         with autocast():
#             seg_out = model(image)
#             loss = criterion(seg_out, seg_truth)
#             losses.update(loss.items(), BATCH_SIZE)
#             losses.update(loss,BATCH_SIZE)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad()
#         if scheduler is not None:
#             scheduler.step()
#     return losses.avg


# def testing_train_loop(
#     image, seg_truth, model, optimizer, criterion, scheduler, device
# ):
#     losses = AverageMeter()
#     scaler = GradScaler()

#     BATCH_SIZE = 2
#     model.train()
#     image = image.to(device).float()
#     seg_truth = seg_truth.to(device)

#     with autocast():
#         seg_out = model(image)
#         loss = criterion(seg_out, seg_truth)
#         losses.update(loss.items(), BATCH_SIZE)
#     scaler.scale(loss).backward()
#     scaler.step(optimizer)
#     scaler.update()
#     optimizer.zero_grad()
#     if scheduler is not None:
#         scheduler.step()
#             scheduler.step(losses.avg)
#     return losses.avg


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     config = {
#         "input_shape": (1, 4, [128, 128, 128]),
#         "n_labels": 3,
#         "activation": "relu",
#         "normalizaiton": "group_normalizaiton",
#     }

#     train_dir = "/nfs/Workspace/brats_brain_segmentation/data/BraTS2021_data/training"

#     train_dataset, val_dataset = get_dataset(train_dir, 1111, fold_num=2)

#     train_loader = DataLoader(
#             train_dataset, batch_size=4, shuffle=True, num_workers=16, pin_memory=True
#         )

#     net = NvNet(config)
#     parameters = params = filter(lambda p: p.requires_grad, net.parameters())
#     optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

#     criterion = SoftDiceLossSquared()
#     print(type(train_loader))
#     print(testing_train_loop( net, optimizer,train_loader, criterion, scheduler, device))
