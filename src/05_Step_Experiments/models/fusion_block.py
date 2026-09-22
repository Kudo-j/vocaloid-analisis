import torch
import torch.nn as nn


class BranchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=in_dim, embedding_dim=out_dim)

    def forward(self, x):
        # x shape: (Batch, Seq, in_dim) - One-Hot vectors
        # Convert to indices: (Batch, Seq)
        indices = torch.argmax(x, dim=-1)
        # Output shape: (Batch, Seq, out_dim)
        return self.embedding(indices)


class BranchCNN1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=9):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch, Seq, in_dim) -> Conv1d expects (Batch, Channels, Seq)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        return x


class BranchCNN2D(nn.Module):
    def __init__(self, in_dim, out_dim, channels=32):
        super().__init__()
        # 2次元畳み込み (周波数 x 時間)
        self.conv1 = nn.Conv2d(
            1, channels, kernel_size=(5, 5), padding=(2, 2), bias=False
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv2 = nn.Conv2d(
            channels, out_dim, kernel_size=(5, 5), padding=(2, 2), bias=False
        )
        self.relu2 = nn.ReLU()
        # 周波数軸を1に集約し、時間軸長は同一に維持
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        # x shape: (Batch, Seq, in_dim) -> Conv2d expects (Batch, 1, in_dim, Seq)
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # (Batch, out_dim, 1, Seq) -> (Batch, Seq, out_dim)
        x = x.squeeze(2).transpose(1, 2)
        return x


class BranchLSTM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 単層の一方向LSTMをデフォルトとする（重くなるのを防ぐため）
        self.lstm = nn.LSTM(in_dim, out_dim, batch_first=True, bidirectional=False)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class FeatureFusionBlock(nn.Module):
    def __init__(self, x_dims_dict, strategy="pass_through", branch_modules=None):
        super().__init__()
        self.strategy = strategy
        self.x_dims_dict = x_dims_dict
        if branch_modules is None:
            branch_modules = {}

        # ==========================================
        # ★ Premodel の内部設定（ハイパーパラメータ）
        # ==========================================
        self.out_dim = 64  # 各ブランチを拡張する次元数
        self.conv_kernel = 9  # CNNを選んだ場合のカーネルサイズ
        # ==========================================

        self.branches = nn.ModuleDict()
        self.output_dim = 0

        if self.strategy == "pass_through":
            self.output_dim = sum(self.x_dims_dict.values())

        elif self.strategy == "late_concat":
            for b_name, b_in_dim in self.x_dims_dict.items():
                b_type = branch_modules.get(b_name, "none")
                self.branches[b_name] = self._build_branch(b_type, b_in_dim)

            # Noneが混ざっている場合でも、concatなら次元数を計算可能
            total_dim = 0
            for b_name, b_in_dim in self.x_dims_dict.items():
                b_type = branch_modules.get(b_name, "none")
                if b_type == "none":
                    total_dim += b_in_dim
                else:
                    total_dim += self.out_dim
            self.output_dim = total_dim

        elif self.strategy == "late_add":
            for b_name, b_in_dim in self.x_dims_dict.items():
                b_type = branch_modules.get(b_name, "linear")
                if b_type == "none" and b_in_dim != self.out_dim:
                    raise ValueError(
                        f"[late_add] Error: Branch '{b_name}' uses 'none' but its dim ({b_in_dim}) != target dim ({self.out_dim})."
                    )
                self.branches[b_name] = self._build_branch(b_type, b_in_dim)
            self.output_dim = self.out_dim
        else:
            raise ValueError(f"Unknown FUSION_STRATEGY: {self.strategy}")

    def _build_branch(self, b_type, in_dim):
        if b_type == "none":
            return nn.Identity()
        elif b_type == "linear":
            return nn.Sequential(nn.Linear(in_dim, self.out_dim), nn.ReLU())
        elif b_type == "cnn1d":
            return BranchCNN1D(in_dim, self.out_dim, self.conv_kernel)
        elif b_type == "cnn2d":
            return BranchCNN2D(in_dim, self.out_dim)
        elif b_type == "embedding":
            return BranchEmbedding(in_dim, self.out_dim)
        elif b_type == "lstm":
            return BranchLSTM(in_dim, self.out_dim)
        else:
            raise ValueError(f"Unknown branch module type: {b_type}")

    def forward(self, x_dict):
        if self.strategy == "pass_through":
            # 順番を固定して結合
            tensors = [x_dict[k] for k in sorted(x_dict.keys())]
            return torch.cat(tensors, dim=-1)

        processed_tensors = []
        # キーをソートして処理順序を保証
        for k in sorted(x_dict.keys()):
            x_val = x_dict[k]
            if k in self.branches:
                x_val = self.branches[k](x_val)
            processed_tensors.append(x_val)

        if self.strategy == "late_concat":
            return torch.cat(processed_tensors, dim=-1)

        elif self.strategy == "late_add":
            # Stack and sum
            stacked = torch.stack(
                processed_tensors, dim=0
            )  # (num_branches, Batch, Seq, Dim)
            return torch.sum(stacked, dim=0)
