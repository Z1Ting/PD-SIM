
import torch.nn as nn
import torch.nn.functional as F
import torch




class BaseNet(nn.Module):
    def __init__(self, feature_depth, num_classes, attention=True, normalize_attn=True):
        super(BaseNet, self).__init__()
        self.attention = attention
        self.feature_depth = feature_depth


        self.conv1 = nn.Conv3d(1, self.feature_depth[0], kernel_size=3)
        self.norm1 = nn.BatchNorm3d(self.feature_depth[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(self.feature_depth[0], self.feature_depth[1], kernel_size=3, stride=2, padding=3)
        self.norm2 = nn.BatchNorm3d(self.feature_depth[1])
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.branch1 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
        self.branch1_norm1 = nn.BatchNorm3d(64)
        self.branch1_relu1 = nn.ReLU(inplace=True)

        self.classify = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.conv3 = nn.Conv3d(self.feature_depth[1], self.feature_depth[2], kernel_size=3, stride=2, padding=3)
        self.norm3 = nn.BatchNorm3d(self.feature_depth[2])
        self.relu3 = nn.ReLU(inplace=True)

        self.branch2 = nn.Conv3d(128, 128, 1)
        self.branch2_norm2 = nn.BatchNorm3d(128)
        self.branch2_relu2 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv3d(self.feature_depth[2], self.feature_depth[3], kernel_size=3, stride=2, padding=3)
        self.norm4 = nn.BatchNorm3d(self.feature_depth[3])
        self.relu4 = nn.ReLU(inplace=True)

        self.branch3 = nn.Conv3d(128, 128, 1)
        self.branch3_norm3 = nn.BatchNorm3d(128)
        self.branch3_relu3 = nn.ReLU(inplace=True)

        self.dense = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, stride=2, padding=0, bias=True)




        self.classify_scroe = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),

            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.norm1(x)
        x = self.relu1(x)

        score = F.adaptive_avg_pool3d(x, (1, 1, 1))

        scores = self.classify(score.flatten(1, -1))

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        l1 = self.branch1(x)
        l1 = self.branch1_norm1(l1)
        l1 = self.branch1_relu1(l1)

        x = self.conv3(l1)
        x = self.norm3(x)
        x = self.relu3(x)
        feature_ = F.adaptive_avg_pool3d(x, (1, 1, 1))


        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        feature_ = self.classify_scroe(feature_.flatten(1, -1)*scores)

        return feature_

class prun(nn.Module):
    def __init__(self, patch_num, num_classes, feature_depth=None):
        super(prun, self).__init__()
        self.patch_num = patch_num
        self.num_classes = num_classes
        if feature_depth is None:
            feature_depth = [32, 64, 128, 128]
        self.patch_net = BaseNet(feature_depth, num_classes)

    def forward(self, inputs):
        score = self.patch_net(inputs)
        return score


if __name__ == "__main__":
    batch_size = 4
    feature_depth= [32, 64, 128, 128, batch_size]
    test = BaseNet(feature_depth, 2, attention = True)
    input_x = torch.randn((batch_size, 1, 21, 21, 21))
    output_x = test(input_x)
    print(output_x.shape)

