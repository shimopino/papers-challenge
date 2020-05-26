# PyTorchでの実装メモ

## GAN

### ResBlock

BigGANのように中間層にResblockを適用してモデルを深くする際、Resblock内での各モジュールの計算順序に注意が必要。
logitsを出力できるようにするため、Resblock内の順伝播は以下の順に実行

1. ReLU(BatchNorm(logits))
2. Upsample --> shortcut connection
3. ReLU(BatchNorm(Conv))
4. Conv
5. residual + shortcut
