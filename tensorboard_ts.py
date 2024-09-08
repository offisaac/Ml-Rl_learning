from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

image_path=r"dataset/cat/cat-4756360_640.jpg"
image_PIL=Image.open(image_path)
image_array=np.array(image_PIL)

# writer=SummaryWriter("logs")
with SummaryWriter("logs") as writer:
    writer.add_image("test",image_array,1,dataformats='HWC')
    for i in range(100):
        writer.add_scalar("y=2x",i,i)#scalar 标量




