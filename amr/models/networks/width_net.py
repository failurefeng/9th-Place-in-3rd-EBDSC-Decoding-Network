import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
from torch.ao.quantization import fuse_modules

class SpectralFeatures(nn.Module):
    """频域特征提取"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv_freq = nn.Sequential(
            nn.Conv1d(in_channels*2, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # 添加自适应池化层统一输出长度
            nn.AdaptiveAvgPool1d(32)  # 确保输出长度为32
        )
        
    def forward(self, x):
        x = x.clone()
        # FFT变换
        x_fft = torch.fft.rfft(x, dim=2)
        x_mag = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)
        # 拼接幅度和相位信息
        x = torch.cat([x_mag, x_phase], dim=1)
        return self.conv_freq(x)  # 输出shape: [batch, 256, 32]

class StatisticalFeatures(nn.Module):
    """统计特征提取"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
    def forward(self, x):
        # 计算统计特征
        mean = x.mean(dim=2, keepdim=True)      # [batch, 2, 1]
        std = x.std(dim=2, keepdim=True)        # [batch, 2, 1]
        max_val = x.max(dim=2, keepdim=True)[0] # [batch, 2, 1]
        min_val = x.min(dim=2, keepdim=True)[0] # [batch, 2, 1]
        kurtosis = torch.mean((x - mean)**4, dim=2, keepdim=True) / (std**4)
        skew = torch.mean((x - mean)**3, dim=2, keepdim=True) / (std**3)
        
        # 拼接所有统计特征 [batch, 12, 1]
        features = torch.cat([mean, std, max_val, min_val, kurtosis, skew], dim=1)
        
        # 扩展到目标长度 [batch, 12, 32]
        features = features.repeat(1, 1, 32)
        
        # 通过卷积层调整通道数 [batch, 128, 32]
        return self.conv(features)
    
class SEBlock(nn.Module):
    """添加SE注意力模块"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    """改进的残差块"""
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        # bottleneck结构
        self.conv1 = nn.Conv1d(in_c, out_c//4, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c//4)
        self.conv2 = nn.Conv1d(out_c//4, out_c//4, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c//4)
        # 修改: 确保conv3输出通道与shortcut匹配
        self.conv3 = nn.Conv1d(out_c//4, out_c, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_c)
        self.se = SEBlock(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c),
                nn.Dropout(0.1)
            )
        
    def forward(self, x):
        # 保存shortcut连接
        residual = self.shortcut(x)
        
        # 主路径
        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))  # 添加conv3和bn3
        
        # SE注意力
        out = self.se(out)
        
        # 残差连接
        out += residual
        return F.gelu(out)  # 最后的激活


class AdaptiveResNet(nn.Module):
    """自适应深度Resnet架构"""
    def __init__(self, in_c, scale=1):
        super().__init__()
        self.scale = scale
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_c, 64, 7, stride=2*scale, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2*scale, padding=1),

            ResBlock(64, 64, stride=1),
            ResBlock(64, 128, stride=2*scale),
            ResBlock(128, 256, stride=2*scale),
            ResBlock(256, 512, stride=2*scale),

            nn.AdaptiveAvgPool1d(32)  # 统一输出长度
        )

    def forward(self, x):
        return self.feature_extractor(x)  # [batch, 512, 32]


class PositionAwareTransformer(nn.Module):
    """位置感知Transformer编码器"""
    def __init__(self, feat_dim, num_heads, num_layers):
        super().__init__()
        # 使用相对位置编码
        self.rel_pos_encoder = nn.Parameter(torch.randn(32, feat_dim))
        # 添加输入层归一化
        self.input_norm = nn.LayerNorm(feat_dim)
        # 层次化Transformer
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=feat_dim,
                    nhead=num_heads,
                    dim_feedforward=2048,
                    dropout=0.2,
                    batch_first=True
                ), 
                num_layers//3
            ) for _ in range(3)
        ])
        # 添加最终层归一化
        self.final_norm = nn.LayerNorm(feat_dim)

    def forward(self, x, data_len):
        # x: [batch, 512, 32]
        x = x.permute(0, 2, 1)  # [batch, 32, 512]

        # 输入归一化
        # x = self.input_norm(x)

        # 生成padding mask
        mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        for i, l in enumerate(data_len):
            valid_len = min(int(l / (2e7 // 32)), 32)  # 根据实际采样率调整,当前为20MHz
            mask[i, valid_len:] = True

        # 添加位置编码
        x = x + self.rel_pos_encoder.unsqueeze(0)

        # 逐层Transformer
        for transformer in self.transformers:
            x = transformer(x, src_key_padding_mask=~mask)

        # 最终归一化
        # x = self.final_norm(x)
        return x.mean(dim=1)  # 全局平均池化

class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels*2, in_channels, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels//16, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels//16, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # x1: [batch, 512, 32]
        # x2: [batch, 512, 32]
        if x1.size(2) != x2.size(2):
            x2 = F.interpolate(x2, size=x1.size(2))
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1x1(x)
        att = self.attention(x)
        return x * att
    
class ImprovedRegressor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        return self.fc_out(x)

class HybridModel(nn.Module):
    """混合信号处理模型"""
    def __init__(self):
        super().__init__()
        # 添加memory efficient配置
        torch.backends.cudnn.benchmark = True  # 启用cuDNN基准测试
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32
        torch.backends.cudnn.allow_tf32 = True
        
        # 添加输入归一化层
        self.input_bn = nn.BatchNorm1d(2)
        
        # 多尺度特征提取
        self.resnet_small = AdaptiveResNet(2, scale=1)
        self.resnet_medium = AdaptiveResNet(2, scale=2)
        self.resnet_large = AdaptiveResNet(2, scale=4)
        
        # 特征融合
        self.fusion = FeatureFusion(512)
        # 添加新的特征提取器
        self.spectral_features = SpectralFeatures(2)
        self.statistical_features = StatisticalFeatures()
        
        # 特征融合层
        self.fusion_all = nn.Sequential(
            nn.Conv1d(512 + 256 + 128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            SEBlock(512)
        )
        
        # 增强型Transformer
        self.transformer = PositionAwareTransformer(feat_dim=512, num_heads=8, num_layers=12)
        
        # 改进的回归头
        self.regressor = ImprovedRegressor(512)
        # 初始化权重
        self._init_weights()
        # 使用混合精度训练
        self.use_amp = True
        

    def _init_weights(self):
        """Kaiming初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if hasattr(m, 'bias') and m.bias is not None:  # 检查bias是否存在
                    nn.init.constant_(m.bias, 0.01)
    

    def _forward_impl(self, x, data_len):
        x = self.input_bn(x)
        # 特征提取
        features_small = self.resnet_small(x) 
        features_medium = self.resnet_medium(x)
        features_large = self.resnet_large(x)

        features_spatial = self.fusion(features_small, features_medium)
        features_spatial = self.fusion(features_spatial, features_large)
        
        # 频域特征
        features_spectral = self.spectral_features(x)
        
        # 统计特征
        features_stats = self.statistical_features(x)
        
        # 融合所有特征
        features_all = torch.cat([
            features_spatial,  # [B, 512, 32]
            features_spectral,  # [B, 256, 32]
            features_stats,    # [B, 128, 32]
        ], dim=1)
        
        features = self.fusion_all(features_all)

        # 时序建模
        temporal_feat = self.transformer(features, data_len)  # [batch, 512]

        # 回归预测
        return self.regressor(temporal_feat).squeeze()
    
    def forward(self, x, data_len):
        # 使用contiguous()优化内存访问
        x = x.contiguous()
        # 减少不必要的数据移动
        if self.use_amp:
            with amp.autocast(device_type='cuda'):
                return self._forward_impl(x, data_len)
        return self._forward_impl(x, data_len)
