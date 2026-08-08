import torch
import torch.nn as nn


class UnifiedModel(nn.Module):
    """
    Premodel (FusionBlock) と メイン予測モデル (core_model) を
    1つのモデルとしてラップするクラス。
    これにより、torch.save や optimizer への登録がシンプルになります。
    """

    def __init__(self, fusion_block, core_model):
        super().__init__()
        self.fusion = fusion_block
        self.core = core_model

    def forward(self, x_dict):
        # 1. 辞書型のデータをFusionブロックに通し、指定された手法で高次元化・結合する
        x_fused = self.fusion(x_dict)

        # 2. 結合されたテンソルをメインモデルに通して予測値を出力する
        return self.core(x_fused)
