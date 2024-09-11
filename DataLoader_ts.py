import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

test_data=torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True,transform=transforms.ToTensor())#"./"其中点代表当前文件夹 /代表下一路径 download可以一直给true

test_dataloader =DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)

print(test_data[0][0].shape)#注意 只有tensor类型下处理的数据才有shape PIL对象是没有的 返回(3,32,32) 3代表3通道(默认红色绿色蓝色) 32代表高度像素数字 32代表宽度像素数字
# with SummaryWriter("batch_logs") as writer:
#     for i, data in enumerate(test_dataloader):
#         if i >=10:
#             break
#         imgs,targets=data
#         img_grid=torchvision.utils.make_grid(imgs)
#         writer.add_image('imgs',img_grid,i)
#         print(imgs.shape)
#         print(targets)

with SummaryWriter("batch_logs") as writer:

    for epoch in range(1,6):
        for i, data in enumerate(test_dataloader):
            if i >=10:
                break
            imgs,targets=data
            img_grid=torchvision.utils.make_grid(imgs)
            writer.add_image(f'Epoch{epoch}',img_grid,i)



