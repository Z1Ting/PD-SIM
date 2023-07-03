# 634-2实验室
# 菜鸟：龚浩强
# 开发时间：2023/3/9 18:54
import torch
import numpy as np


class NTXentLoss_valid(torch.nn.Module):
        # NTXentLoss(self.device, self.config['batch_size'], **self.config['loss'])
    def __init__(self, device, batch_size, temperature_or_m, use_cosine_similarity):
        super(NTXentLoss_valid, self).__init__()
        self.batch_size = 1
        self.temperature = temperature_or_m
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        # return mask.to(self.device)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        # return self._cosine_simililarity
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        # 生成
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)  ？？？？2N？  不是N？
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        # self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        return v

    def forward(self, zis, zjs):
        # 按维数0（行）拼接
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        # 从正面样本中过滤出分数
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        # 把其他全当负类  两倍的batch_size  所以这里不需要标签
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        # print(loss)
        # o1 = torch.nn.LogSoftmax(dim=1)
        # o2 = torch.nn.NLLLoss(reduction='none')
        # p1 = o1(logits)
        # print(p1)
        # p2 = o2(p1, labels)
        # print(p2)

        return loss / (2 * self.batch_size)


def main():
    Loss = NTXentLoss_valid('cuda', 3, 0.5, True)
    print(Loss.mask_samples_from_same_repr)
    testi = np.array([
        [0.5, 0.5, 0],
        [0.2, 0.4, 0.2],
        [0.1, 0.8, 0.1]
    ])

    testj = np.array([
        [0.5, 0.5, 0],
        [0.2, 0.4, 0.2],
        [0.1, 0.8, 0.1]
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testi.to(device)
    testj.to(device)

    xi = torch.tensor(testi)
    xj = torch.tensor(testj)
    loss = Loss(xi, xj)
    print(loss)


if __name__ == "__main__":
    main()
