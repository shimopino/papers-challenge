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

You can run VSCode Server on Colab notebook.

- [Colab on steroids: free GPU instances with SSH access and Visual Studio Code Server](https://towardsdatascience.com/colab-free-gpu-ssh-visual-studio-code-server-36fe1d3c5243)
- [ngrok](https://dashboard.ngrok.com/get-started/setup)