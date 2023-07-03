import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from resnet_3D import generate_model
import torch

import yaml

mid = open("G:\Experience\SimCLR-pytorch-master\config.yaml", encoding='utf-8')
config = yaml.load(mid, Loader=yaml.FullLoader)

class ResNetSimCLR_3D(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_3D, self).__init__()
        # self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
        #                     "resnet50": models.resnet50(pretrained=False)}
        self.resnet = {"resnet18": generate_model(18),
                       "resnet50": generate_model(50)}
        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features
        # print(num_ftrs)
        resnet_list = list(resnet.children())
        # del resnet_list[0]  # 删除对象 删除空间注意力模块  这行代码注释同时需要注释下面那行
        # resnet_list[0] = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)  #第一个参数原为3channels，灰质图像用1channel
        resnet_list[1] = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)  #第一个参数原为3channels，灰质图像用1channel



        # 参考论文 B.9 修改第一个卷积层并去掉第一个max pooling并去掉最后一层
        self.features = nn.Sequential(*resnet_list[:-1])  # 重建了一个resnet网络
        '''后续classifier要取的features结构，用于分类器的训练'''
        # projection MLP
        mid_out = num_ftrs // 2
        self.l1 = nn.Linear(num_ftrs, mid_out)
        self.l2 = nn.Linear(mid_out, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        # print("input_shape:")
        # print(x.shape)
        h = self.features(x)
        # print(self.features)
        # print(h.shape)

        h1 = h.squeeze()
        # print(h.shape)

        # print("h_shape:")
        # print(h.shape)
        # print(h.shape)
        # print(1111)

        # 后加 把32*512 变为 32*2
        # fc2 = nn.Linear(512, 128).cuda()
        # fc3 = nn.Linear(128, 2).cuda()
        # softmax = nn.Softmax(dim=1).cuda()
        # h2 = fc2(h)
        # h2 = fc3(h2)
        # h2 = softmax(h2)
        # print(h2.shape)

        # 加sigmoid
        # fc2 = nn.Linear(512, 2).cuda()
        # h2 = fc2(h)
        # sigmoid = nn.Sigmoid().cuda()
        # h2 = sigmoid(h2)


        x = self.l1(h1)
        x = F.relu(x)
        x = self.l2(x)
        # print("x_shape:")
        # print(x.shape)
        return h1, x


if __name__ == "__main__":
    test = ResNetSimCLR_3D('resnet18', 128)
    print(test)
    input_x = torch.randn((16, 1,  62, 62, 62)) # 第1维由自己定义的resnet的n_input_channels等于几以及     ((10, 1, 2, 2, 2))
                                            # 这里的resnet_list[0] = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  #第一个参数原为3channels，灰质图像用1channel（这里可以绝对决定）
                                            # 一起决定，这里设置为1，因为是灰质图象
    output_x1,output_x2 = test(input_x)
    print(output_x1.shape, output_x2.shape)
