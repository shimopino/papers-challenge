# Colab Tips

## How to Use Full Session [12 hours]

run this code on developer console.

```js
function ClickButton() {
    console.log("Working");
    document.querySelector("colab-toolbar-button").click();
}
setInterval(ClickButton, 60000) // set 60,000 millisecond
```

- [Google Colab Tips for Power Users](https://madewithml.com/projects/1609/google-colab-tips-for-power-users/)

## VSCode Server on Colab

### User Pure ngrok

You can run VSCode Server on Colab notebook.

```python
# Install useful stuff
! apt install --yes ssh screen nano htop ranger git > /dev/null

# SSH setting
# 以下で設定したパスワードを、SSHアクセス時に使用する
! echo "root:carbonara" | chpasswd
! echo "PasswordAuthentication yes" > /etc/ssh/sshd_config
! echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config
! echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
! service ssh restart > /dev/null

# Download ngrok
! wget -q -c -nc https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
! unzip -qq -n ngrok-stable-linux-amd64.zip

# Run ngrok
authtoken = "PUT_YOUR_TOKEN_HERE"
get_ipython().system_raw('./ngrok authtoken $authtoken && ./ngrok tcp 22 &')
! sleep 3

# Get the address for SSH
import requests
from re import sub
r = requests.get('http://localhost:4040/api/tunnels')
str_ssh = r.json()['tunnels'][0]['public_url']
str_ssh = sub("tcp://", "", str_ssh)
str_ssh = sub(":", " -p ", str_ssh)
str_ssh = "ssh root@" + str_ssh
print(str_ssh)
```

google drive mount

```python
# Mount Google Drive and make some folders for vscode
from google.colab import drive
drive.mount('/googledrive')
! mkdir -p /googledrive/My\ Drive/colabdrive
! mkdir -p /googledrive/My\ Drive/colabdrive/root/.local/share/code-server
! ln -s /googledrive/My\ Drive/colabdrive /
! ln -s /googledrive/My\ Drive/colabdrive/root/.local/share/code-server /root/.local/share/
```

install VSCode

```python
! curl -fsSL https://code-server.dev/install.sh | sh > /dev/null
! code-server --bind-addr 127.0.0.1:9999 --auth none &
```

ssh access

```python
# password: carbonara
ssh -L 9999:localhost:9999 root@0.tcp.ngrok.io -p 14407
```

- [Colab on steroids: free GPU instances with SSH access and Visual Studio Code Server](https://towardsdatascience.com/colab-free-gpu-ssh-visual-studio-code-server-36fe1d3c5243)
- [chezou/code-server.ipynb](https://gist.github.com/chezou/858d663381625c9bb1c868e0c95969c6)
- [ngrok](https://dashboard.ngrok.com/get-started/setup)

### Use third-party library

```python
!pip install colabcode

from colabcode import ColabCode
ColabCode(port=10000, password="sample")a
```

## External Data

- [外部データ: ローカル ファイル、ドライブ、スプレッドシート、Cloud Storage](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=F1-nafvN-NwW)
