# 環境回り

## Docker

### Anaconda周り

Anacondaでパッケージをインストールする際にキャッシュを消去することでImageのサイズを小さくすることができる。

```sh
$ conda clean -ya
```

### pip周り

pipでパッケージをインストールする際にキャッシュを保存しないようにすることでDockerImageのサイズを小さくすることができる。

```sh
$ pip3 install --no-cache-dir <package>
```

- [参考](https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for)

### 容量解消

以下のコマンドでDockerがどの程度容量を占有しているのかがわかる。

```sh
$ docker system df
```

ローカルディスクの容量がの固定た場合には以下のように削除する。

```sh
$ docker system prune -a --volumnes
```

- [Docker for Macを使っていたら50GB位ディスク容量を圧迫していたのでいろんなものを削除する](https://qiita.com/shinespark/items/526b70b5f0b1ac643ba0)

## Docker-Compose

### error exited with code 0

`Dockerfile`に特に実行するコマンドを指定していない場合には、コンテナを起動したとしても実行がすぐに終了してしまう。
この現象を防ぐには`docker-compose.yml`に`tty`を設定する。

```docker
version: "3"
services:
  tensorboard:
    tty: true
```

### tensorboard

`Docker-Compose`を使用してTensorboardを起動するには`Dockerfile`に`ENTRYPOINT`を指定してコマンドを記載する方法か、`docker-compose.yml`に直接コマンドを記載する。

```docker
version: "3"
services:
  tensorboard:
    image: "image name"
    container_name: "container name"
    build:
      context: .
      dockerfile: "Dockerfile name"
      volumnes:
        - /host/logs:/logs
      ports:
        - 6006:6006
      command: tensorboard --logdir=/logs --bind-all
```

## Tensorboard

### コンテナ上のTensorboardに外部アクセス

Dockerコンテナ内で以下のようにTensorboardを起動すると外部からアクセスすることができない。
（VirtualBoxを利用するDocker Toolboxの現象かもしれない）

```sh
(container) $ tensorboard --logir=/path/to/logs

(host) $ curl localhost:6006
```

コマンド実行時にすべてのNICに対してポートを指定することで別ホストからでもアクセスすることが可能となる。

```sh
# "--bind-all"フラグをつける
(container) $ tensorboard --logdir=/path/to/logs --bind-all

# "--host=0.0.0.0"を指定する
(container) $ tensorboard --logdir=/path/to/logs --host=0.0.0.0
```

## JupyterLab

- [JupyterLabのおすすめ拡張機能8選](https://qiita.com/canonrock16/items/d166c93087a4aafd2db4)

 