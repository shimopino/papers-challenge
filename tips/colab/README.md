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