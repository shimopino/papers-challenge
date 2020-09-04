# Remote Development機能を活用した開発環境の構築手順をまとめる

## 準備

ローカルの開発環境には、Dockerがインストールされている必要がある。

## 基本的な接続方法

リモート端末で起動しているDockerコンテナに、VSCodeをAttachして開発を行う際には、単純に`.vscode`フォルダ配下の`settings.json`にて、以下の設定を追加して、VSCodeを再起動すればい。

```json
"docker.host":"ssh://your-remote-user@your-remote-machine-fqdn-or-ip-here"
```

接続には鍵認証方式でのSSH接続を行うため、予め公開鍵・秘密鍵の準備は行っておく。

次にリモート上で起動するDockerの設定を行う。
ポイントとしては、コンテナの設定を記述する`devcontainer.json`はリモート端末上に配置する必要がある点である。

この際に、以下のような設定を行い、リモート端末上で可動しているDockerコンテナが、リモートに存在するファイルシステムにマウントを取っておくことで、リモートのファイルに直接アクセスすることなく編集を行うことができる。

```json
{
  "image": "node", // Or "dockerFile"
  "workspaceFolder": "/workspace",
  "workspaceMount": "source=remote-workspace,target=/workspace,type=volume"
}
```

これは、コマンド`Remote-Containers: Open Repository in Container...`を実行しても同じ挙動になる`a。

## 参照記事

- [Developing inside a container on a remote Docker host](https://code.visualstudio.com/docs/remote/containers-advanced#_developing-inside-a-container-on-a-remote-docker-host)
- [VS Code Development Using Docker Containers on Remote Host](https://leimao.github.io/blog/VS-Code-Development-Remote-Host-Docker/)
