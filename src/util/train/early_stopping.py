"""
early_stopping.py
=================
学習の早期終了 (Early Stopping) を管理するクラス。

検証Lossが一定エポック数改善しなかった場合に学習を停止し、
不要な計算時間を削減する。Optunaでの大量試行時に特に有効。
"""


class EarlyStopping:
    """
    Early Stopping の管理クラス。

    Args:
        patience: 改善が見られないまま許容するエポック数。
        min_delta: 改善とみなす最小の変化量。
    """

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        """
        検証Lossを受け取り、早期終了すべきかを判定する。

        Args:
            val_loss: 現在のエポックの検証Loss値。

        Returns:
            True: 学習を停止すべき場合。
            False: 学習を継続すべき場合。
        """
        if val_loss < self.best_loss - self.min_delta:
            # 改善が見られた場合: カウンタをリセット
            self.best_loss = val_loss
            self.counter = 0
        else:
            # 改善なし: カウンタを進める
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(
                    f"[EarlyStopping] {self.patience}エポック改善なし。学習を停止する。"
                )

        return self.should_stop
