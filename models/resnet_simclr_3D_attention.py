import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from resnet_3D_attention import generate_model
import torch

import yaml

mid = open("G:\Experience\SimCLR-pytorch-master\simclr_attention\config_attention.yaml", encoding='utf-8')
config = yaml.load(mid, Loader=yaml.FullLoader)


class AttentionBlock(nn.Module):
    def __init__(self, ):
        super(AttentionBlock, self).__init__()


    def forward(self, input, patch_pred):  # attn_feat, ca = self.attention_net(feature_maps, patch_scores)  # return mean_input*a, a.flatten(1)
        mean_input = input.mean(2)  # mean(input, axis=2)  torch.Size([16, 60, 128, 6, 6, 6])
        patch_pred = patch_pred.unsqueeze(-1)
        patch_pred = patch_pred.unsqueeze(-1)
        patch_pred = patch_pred.unsqueeze(-1)  # 连续三次在最后一维增加一个维度
        a = mean_input + patch_pred  #
        a = torch.sigmoid(a)  # torch.Size([1, 60, 1, 1, 1])
        # print(a.shape)
        return mean_input*a, a.flatten(1)  # a.flatten(1)应该是输出注意力图的



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
        # resnet_list = list(resnet.children())
        # del resnet_list[3]  # 删除对象 删除列表第四个元素
        # resnet_list[0] = nn.Conv3d(config['resnet_floor1_input_num'], 64, kernel_size=3, stride=1, padding=1, bias=False)  #第一个参数原为3channels，灰质图像用1channel
        # 参考论文 B.9 修改第一个卷积层并去掉第一个max pooling并去掉最后一层
        # self.features = nn.Sequential(*resnet_list[:-1])  # 重建了一个resnet网络
        self.features = resnet
        self.attention_net = AttentionBlock()
        self.bn = nn.BatchNorm3d(2)

        '''后续classifier要取的features结构，用于分类器的训练'''

        # self.mlp = nn.Sequential(
        #     nn.BatchNorm3d(2),
        #     nn.Conv3d(2, 16, kernel_size=1),
        #     nn.ReLU(True),
        #     # torch.nn.Dropout(p=0.5),  #new
        #     # nn.Linear(16, 2),
        #

        # )
        # projection MLP
        mid_out = num_ftrs // 2
        self.l1 = nn.Linear(7, mid_out)

        # self.l1 = nn.Linear(num_ftrs, mid_out)
        self.l2 = nn.Linear(mid_out, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, inputs):
        # print("input_shape:")
        # print(inputs.shape)  # torch.Size([30, 16, 1, 25, 25, 25])
        # print(inputs[0].shape)  # torch.Size([16, 1, 25, 25, 25])
        patch_feature, patch_score = [], []
        # print(len(inputs))
        for i in range(len(inputs)):  # patch_num = 30
            # print(i)
            feature, score = self.features(inputs[i])
            # RuntimeError: mat1 and mat2 shapes cannot be multiplied(640000x25 and 64x1)

            feature = feature.unsqueeze(1)
            patch_feature.append(feature)
            patch_score.append(score)


        feature_maps = torch.cat(patch_feature, 1)  # torch.Size([8, 60, 128, 11, 11, 11])
        # print(feature_maps.shape)
        patch_scores = torch.cat(patch_score, 1)  # torch.Size([8, 64, 1, 1, 1])
        # print(patch_scores.shape)

        attn_feat1, ca = self.attention_net(feature_maps, patch_scores)  # return mean_input*a, a.flatten(1)
        # print('attn_feat.shape')
        b, c, h, w, l = attn_feat1.shape
        # print(attn_feat.shape)  # torch.Size([16, 30, 1, 1, 1])
        # attn_feat_mlp = self.mlp(attn_feat)
        # attn_feat_mlp = attn_feat + attn_feat_mlp


        # print(attn_feat[0].shape)
        # print(len(attn_feat))

        # print(len(attn_feat[0]))
        # s = 0
        # for i in range(len(attn_feat)):
        #     for j in range(len(attn_feat[0])):
        #         single_patch_feat = attn_feat[i][j].unsqueeze(0)
        #         print(single_patch_feat.shape)
        #         s += 1
        #         print(s)

        attn_feat2 = attn_feat1.squeeze()
        # print(attn_feat2.shape)
        # attn_feat_mlp = attn_feat_mlp.squeeze()

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


        attn_feat_mlp = self.l1(attn_feat2)
        attn_feat_mlp = F.relu(attn_feat_mlp)
        attn_feat_mlp = self.l2(attn_feat_mlp)
        # print("x_shape:")
        # print(x.shape)
        return attn_feat1, attn_feat_mlp


if __name__ == "__main__":
    test = ResNetSimCLR_3D('resnet18', 128)

    input_x = torch.randn((2, 16, 1, 25, 25, 25)) # 第1维由自己定义的resnet的n_input_channels等于几以及     ((10, 1, 2, 2, 2))
                                            # 这里的resnet_list[0] = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  #第一个参数原为3channels，灰质图像用1channel（这里可以绝对决定）
                                            # 一起决定，这里设置为1，因为是灰质图象
    output_x1,output_x2 = test(input_x)
    print(output_x1.shape, output_x2.shape)
