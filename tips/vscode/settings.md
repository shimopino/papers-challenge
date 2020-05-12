# VSCodeの主要設定

再現実装時に使用しているVSCode周りの環境設定をいかに記載

まずDockerfileの定義は以下のようにPyTorchを読み込む形にしておく。
またLinterとして`flake8`、Formatterとして`black`を使用する。

```Dockerfile
# https://github.com/microsoft/vscode-dev-containers/tree/v0.112.0/containers/python-3/.devcontainer/base.Dockerfile
ARG VARIANT="3"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

RUN apt-get update \
   && apt-get -y install --no-install-recommends git \
   && apt-get autoremove -y \
   && apt-get clean -y \
   && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir flake8 black
RUN pip install --no-cache-dir torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

このDockerfileを`.devcontainer`以下に配置しておき、`devcontainer.json`で指定する。
設定ファイルにはLinterやFormatterの記述もしておく。

```json
{
	"name": "Python 3",
	"build": {
		"dockerfile": "Dockerfile",
		"context": "..",
		// 以下はdocker image buildの際に渡される引数を指定する
		"args": {
			"VARIANT": "3"
		}
	},
	// 以下はdocker runの際に渡される引数を指定する
	"runArgs": [
		// 以下でnvidia dockerを起動できるようにする
		"--gpus=all"
	]
	"settings": {
		"terminal.integrated.shell.linux": "/bin/bash",
   		// PythonのPathを指定。Anacondaの場合には/opt/conda/bin/python
		"python.pythonPath": "/usr/local/bin/python",
    		// Lintingの設定は以下
		"python.linting.enabled": true,
		"python.linting.pylintEnabled": false,
		"python.linting.flake8Enabled": true,
		"python.linting.lintOnSave": true,
    		// Formatterには"black"を指定しておき、保存時に自動的に適用する
		"python.formatting.provider": "black",
		"editor.formatOnSave": true
	},
	"extensions": [
		"ms-python.python",
    		// 以下の拡張機能は必須
		"tabnine.tabnine-vscode",
		"coenraads.bracket-pair-colorizer-2",
		"christian-kohler.path-intellisense"
	]
}
```

VSCodeを開いているフォルダ上に`main.py`が存在しているとき、デバッグの構成は以下のように行う。

```json
{
  // Use IntelliSense to learn about possible attributes.
  // Hover to view descriptions of existing attributes.
  // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
  "version": "0.2.0",
  "configurations": [
    {
      "name": "sample debug",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "console": "integratedTerminal"
    }
  ]
}
```

これでPyTorchなどでデバッグが可能となる。
