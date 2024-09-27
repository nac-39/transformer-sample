import torch
from transformer_sample.transformer import Transformer

# ハイパーパラメータの設定
src_vocab_size = 10000  # ソース語彙サイズ
trg_vocab_size = 10000  # ターゲット語彙サイズ
src_pad_idx = 0         # ソースパディングインデックス
trg_pad_idx = 0         # ターゲットパディングインデックス
embed_size = 256        # 埋め込みサイズ
num_layers = 6          # レイヤー数
forward_expansion = 4   # フォワード拡張
heads = 8               # ヘッド数
dropout = 0.1           # ドロップアウト率
device = "cuda"         # デバイス
max_length = 100        # 最大長

# Transformerモデルの初期化
model = Transformer(
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    trg_pad_idx,
    embed_size,
    num_layers,
    forward_expansion,
    heads,
    dropout,
    device,
    max_length,
).to(device)

# モデルのテスト入力
src = torch.randint(0, src_vocab_size, (32, max_length)).to(device)  # バッチサイズ32のランダムなソース入力
trg = torch.randint(0, trg_vocab_size, (32, max_length)).to(device)  # バッチサイズ32のランダムなターゲット入力

# フォワードパス
output = model(src, trg)
print(output.shape)  # 出力の形状を表示
