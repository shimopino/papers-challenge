# PyTorchでの実装メモ

## 高速化などなど

- データ読み込み時の工夫

```python
# 非同期読込のために num_workers > 0 かつ pin_memory=True がいい
config = {'num_workers': 1, 'pin_memory': True}
```

- 非同期でのCPU-GPU間通信

```python
for data, target in loader:
    # Overlapping transfer if pinned memory
    data = data.to('cuda:0', non_blocking=True)
    target = target.to('cuda:0', non_blocking=True)

    # The following code will be called asynchronously,
    # such that the kernel will be launched and returns control 
    # to the CPU thread before the kernel has actually begun executing
    output = model(data)  # has to wait for data to be pushed onto device (synch point)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

- 決定論的な挙動ではなくなるが、cuDNNの最適化を有効にする。

```python
torch.backends.cudnn.benchmark = True
```

- バッチ正規化層の前の畳み込み層ではバイアスを使用しない

```python
# no
layer = nn.Sequential(
    nn.Conv2d(..., bias=True),
    nn.BatchNorm2d(...)
)

# yes
layer = nn.Sequential(
    nn.Conv2d(..., bias=False),
    nn.BatchNorm2d(...)
)
```

- 勾配初期化時に、書き込みのみを行い

```python
# これは(read + write)な挙動
model.zero_grad()

# これは(write)のみの挙動
for param in model.parameters():
    param.grad = None
```

- 複数のGPUを使用する際は、プロセスの並列に実行させる

```python
# no
DataParallel

# yes
DistributedDataParallel
```

- apexを使う

- 順伝播時に必要な演算に関する出力のみを保存する

```python
# activation (ReLU, Sigmoid, ...)
# Up / Down Sampling
# Matrix-vector ops with small accumulation depth
torch.utils.checkpoint
```

### 参考文献

- [PyTorch Performance Guide](https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/)
- [https://twitter.com/karpathy/status/1299921324333170689](https://twitter.com/karpathy/status/1299921324333170689)

## GAN

### ResBlock

BigGANのように中間層にResblockを適用してモデルを深くする際、Resblock内での各モジュールの計算順序に注意が必要。
logitsを出力できるようにするため、Resblock内の順伝播は以下の順に実行

1. ReLU(BatchNorm(logits))
2. Upsample --> shortcut connection
3. ReLU(BatchNorm(Conv))
4. Conv
5. residual + shortcut
