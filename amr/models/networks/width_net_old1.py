import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """稳健型残差块，自动处理维度变化"""
    def __init__(self, in_c, out_c, stride=2):
        """
        :param in_c: 输入的通道数
        :param out_c: 输出的通道数
        :param stride: 滑动步长
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class AdaptiveResNet(nn.Module):
    """自适应深度Resnet架构"""
    def __init__(self, in_c):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_c, 64, 7, 2, 3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1),

            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),

            nn.AdaptiveAvgPool1d(32)  # 统一输出长度
        )

    def forward(self, x):
        return self.feature_extractor(x)  # [batch, 512, 32]


class PositionAwareTransformer(nn.Module):
    """位置感知Transformer编码器"""
    def __init__(self, feat_dim, num_heads, num_layers):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.randn(1, 32, feat_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, data_len):
        # x: [batch, 512, 32]
        x = x.permute(0, 2, 1)  # [batch, 32, 512]

        # 生成padding mask
        mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        for i, l in enumerate(data_len):
            valid_len = min(int(l / (2e7 // 32)), 32)  # 根据实际采样率调整,当前为20MHz
            mask[i, valid_len:] = True

        x = x + self.pos_encoder
        x = self.transformer(x, src_key_padding_mask=~mask)
        return x.mean(dim=1)  # 全局平均池化


class HybridModel_width(nn.Module):
    """混合信号处理模型"""
    def __init__(self):
        super().__init__()
        # 添加输入归一化层
        self.input_bn = nn.BatchNorm1d(2)
        self.resnet = AdaptiveResNet(2)
        self.transformer = PositionAwareTransformer(512, 8, num_layers=6)
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Kaiming初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x, data_len):
        x = self.input_bn(x)
        # 特征提取
        features = self.resnet(x)  # [batch, 512, 32]
        # 时序建模
        temporal_feat = self.transformer(features, data_len)  # [batch, 512]
        # 回归预测
        return self.regressor(temporal_feat).squeeze()