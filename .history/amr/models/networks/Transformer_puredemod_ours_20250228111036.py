from torch.autograd import Variable
from torch.autograd import Variable
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func
from torch.cuda.amp import autocast
import numpy as np

class FlashAttention(nn.Module):
    """兼容flash_attn_func的多头注意力包装"""
    def __init__(self, embed_dim, num_heads, causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        # 定义QKV投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, q, k, v, key_padding_mask=None):
        # 临时转换输入到 FP16
        q, k, v = [x.half() for x in [q, k, v]]

        # 将权重参数转换到 FP16（临时转换，原始参数仍保持 FP32）
        q = F.linear(q, self.q_proj.weight.half(), self.q_proj.bias.half() )
        k = F.linear(k, self.k_proj.weight.half(), self.k_proj.bias.half() )
        v = F.linear(v, self.v_proj.weight.half(), self.v_proj.bias.half() )
        
        batch_size, seq_len, _ = q.shape
        # 分割头 [B, S, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 处理 key_padding_mask（转换为 FlashAttention 所需的格式）
        if key_padding_mask is not None:
            # padding位置的掩码值设为极小值（需匹配数据类型）
            key_padding_mask = key_padding_mask.half().masked_fill(
                key_padding_mask, 
                torch.finfo(q.dtype).min
            )

        # 直接调用flash_attn_func
        attn_out = flash_attn_func(
            q, k, v,
            causal=self.causal,
            softmax_scale=1.0 / (self.head_dim ** 0.5)
        ).float()  # (B, S, num_heads, head_dim)
        
        # 合并多头并输出
        attn_out = attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(attn_out)

# 编码器层（集成Flash Self-Attention）
class FlashEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.2):
        super().__init__()
        # 使用自定义FlashAttention层（无因果）
        self.self_attn = FlashAttention(d_model, num_heads, causal=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, src,  src_key_padding_mask=None):
        # 所有计算在 FP32 下进行，FlashAttention 层内部自动转换
        src = src
        attn_out = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + attn_out)
        # FFN（FP32输入 > FP16输出）
        ffn_out = self.ffn(src)
        return self.norm2(src + ffn_out)

class PositionalEncoding(nn.Module):
    """支持batch_first=True的通用位置编码"""
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model//2,)
        
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 广播至(max_len, d_model//2)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 注册为(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor 形状 (B, S, E)
              B - 批量大小
              S - 序列长度
              E - 特征维度
        """
        # 自动截取与当前序列匹配的编码并扩展维度
        x = x + self.pe[:x.size(1)].unsqueeze(0)  # (1, S, E) → (B, S, E) via广播
        return x

class DynamicEmbeddingNetwork(nn.Module):
    def __init__(self, embed_dim, conv_channel):
        super().__init__()
        self.embed_dim = embed_dim
        # 卷积模块处理每个符号的IQ时间序列
        self.conv_block = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),  # 处理(B', 2, Width_max), B'= B*Num_max
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1), # (B', _, Width_max/2)
            nn.Conv1d(64, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, conv_channel, kernel_size=3, padding=1), 
            nn.BatchNorm1d(conv_channel),
            nn.ReLU(),
        )
        
        # 残差路径卷积（输入2→conv_channel通道）
        self.res_conv = nn.Conv1d(2, conv_channel, kernel_size=4, stride=2, padding=1) # (B', conv_channel, Width_max/2)
        
        # 注意力池化层（时序维度加权）
        self.attention_pool = nn.Sequential(
            nn.Conv1d(conv_channel, conv_channel, kernel_size=1),  
            nn.Softmax(dim=-1)      # 在最后一个维度上归一化
        )
 
        # 符号特征投影
        self.fc = nn.Linear(conv_channel, embed_dim)

    def forward(self, x):
        """
        输入:
            x:        形状 (B, 2, Num_max, Width_max) 的填充张量
            symbol_mask: 形状 (B, Num_max) 的符号掩码（True表示填充）
        输出:
            embeddings:     (B, Num_max, E) 
            padding_mask:   (B, Num_max) （直接返回输入的symbol_mask）
        """
        B, _, Num_max, Width_max = x.shape
        
        # === 步骤1：将批量转换为符号级视角 ===
        # 重组张量将符号维度合并到批次 (B,2,Num_max,Width_max) → (B*Num_max,2,Width_max)
        x_symbols = x.permute(0, 2, 1, 3).reshape(B*Num_max, 2, Width_max)
        
        # === 步骤2：并行处理所有符号 ===
        # 卷积特征提取 
        x_conv = self.conv_block(x_symbols)  # (B*Num_max, conv_channel, Width_max/2)
        
        # === 步骤3：残差连接 ===
        # 残差路径 
        residual = self.res_conv(x_symbols)
        x_residual = x_conv + residual  # (B*Num_max, conv_channel, Width_max/2)

        # === 步骤4：时序注意力池化 ===
        attn_weights = self.attention_pool(x_residual)  # (B*Num_max, conv_channel, Width_max/2)
        x_pooled = torch.sum(x_residual * attn_weights, dim=-1)  # (B*Num_max, conv_channel)
        
        # === 步骤5：特征投影 ===
        
        x_embed = self.fc(x_pooled)  # (B*Num_max, E), 
        
        # === 步骤6：恢复批量视角 ===
        embeddings = x_embed.view(B, Num_max, -1)  # (B,Num_max,E)
        
        # === 输出直接使用传入的符号掩码 ===
        return embeddings




class Transformer_puredemod(nn.Module):
    """全流程batch_first=True架构"""
    def __init__(self, 
                 conv_channel=64, 
                 embed_dim=64, 
                 num_heads=4,
                 num_encoder_layers=8, 
                 dim_feedforward=512, 
                 num_codes=32):
        super().__init__()
        self.embedding_net = DynamicEmbeddingNetwork(embed_dim,conv_channel)
        self.pos_encoder = PositionalEncoding(embed_dim)

        # Transformer参数调整: batch_first=True
        # 使用改进的编码层和解码层
        self.encoder_layers = nn.ModuleList([
            FlashEncoderLayer(embed_dim, num_heads, dim_feedforward)
            for _ in range(num_encoder_layers)
        ])
        self.output_fc = nn.Sequential(
                            nn.Linear(embed_dim, dim_feedforward),
                            nn.LayerNorm(dim_feedforward),
                            nn.GELU(),  # 替换为GELU ★
                            nn.Dropout(0.3),                  # 对应位置新增Dropout
                            nn.Linear(dim_feedforward, num_codes)
)

        self.register_buffer('weight_matrix', torch.linspace(0, num_codes-1, num_codes, dtype=torch.float32))




    def forward(self, pad_splits, split_mask):
        with autocast():
            # 特征提取 (B,Num_max,E)
            src = self.embedding_net(pad_splits)
            src_key_padding_mask = split_mask
            if torch.isnan(src).any():
                print("Nan detected in src after embedding")
            # 位置编码 (保持batch_first)
            src = self.pos_encoder(src)  # (B,Num_max,E)
            if torch.isnan(src).any():
                print("Nan detected in src after pos encoding")

            # # 修改编码器检查点调用（仅传递必须参数）
            memory = src
            for layer in self.encoder_layers:
                memory = layer(
                    memory, 
                    src_key_padding_mask=src_key_padding_mask,
                )
            if torch.isnan(memory).any():
                print("Nan detected in memory")

            tgt = memory
            # 输出映射 (B,Num_max,E)→(B,Num_max,num_codes)
            logits = self.output_fc(tgt)
            if torch.isnan(logits).any():
                print("Nan detected in logits")
            # 转换为数值预测 (B,Num_max)
            output_probs = F.softmax(logits, dim=-1)
            if torch.isnan(output_probs).any():
                print("Nan detected in output_probs")
            preds = torch.sum(output_probs * self.weight_matrix, dim=-1)
            if torch.isnan(preds).any():
                print("Nan detected in preds")
        return preds




