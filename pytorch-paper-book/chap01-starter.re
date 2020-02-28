= Re:VIEW Starterの独自機能

//abstract{
素のRe:VIEWと比べて、Re:VIEW Starter（以下「Starter」とする）はいろんな機能強化やバグフィックスをしています。
この章では、それらについて解説します@<fn>{jmoh7}。

また本章ではRe:VIEWの機能や@<LaTeX>{}の用語が説明なく使われます。
これらが分からなければ読み飛ばして、分かる箇所だけ読んでください。
//}
//footnote[jmoh7][本章では、Re:VIEWや@<LaTeX>{}についての説明はしません。Re:VIEWの書き方についてはマニュアル（@<href>{https://github.com/kmuto/review/blob/master/doc/format.ja.md}）やチートシート（@<href>{https://qiita.com/froakie0021/items/b0f4ba5f242bbd571d4e}）を参照してください。]

#@#//makechaptitlepage[toc=section]


=={sec-ext} 原稿本文を書くための機能


=== 強調

 * 「@<code>$@<nop>{@}<b>{...}$」は、明朝体のまま太字になります（Re:VIEWのデフォルト）。
 * 「@<code>$@<nop>{@}<em>{...}$」は、ゴシック体になります（Starter拡張）。
 * 「@<code>$@<nop>{@}<strong>{...}$」は、太字のゴシック体になります（Starter拡張）。
 * 「@<code>$@<nop>{@}<B>{...}$」は、「@<code>$@<nop>{@}<strong>{...}$」のショートカットです（Starter拡張）。

//emlist[サンプル]{
いろはfghij  @<letitgo>$@$<b>{いろはfghij}  @<letitgo>$@$<em>{いろはfghij}  @<letitgo>$@$<strong>{いろはfghij}
@<letitgo>$@$<B>{いろはfghij}
//}

//sampleoutputbegin[表示結果]

いろはfghij  @<b>{いろはfghij}  @<em>{いろはfghij}  @<strong>{いろはfghij}
@<B>{いろはfghij}

//sampleoutputend



なおソースコードの一部を太字にしたいときは、「@<code>$@<nop>{@}<strong>{...}$」ではなく「@<code>$@<nop>{@}<b>{...}$」を使ってください。
なぜなら「@<code>$@<nop>{@}<strong>{...}$」だとゴシック体になってしまうのに対し、「@<code>$@<nop>{@}<b>{...}$」だとタイプライタ体のまま太字になるからです。

//emlist[サンプル]{
@<letitgo>$//$emlist{
タイプライタ体（通常）： 0123456789 ijkl IJKL !"$%&()*,-./:;?@[\]`|~
タイプライタ体（太字）： @<letitgo>$@$<b>{0123456789 ijkl IJKL !"$%&()*,-./:;?@[\\]`|~}
ゴシック体　　（太字）： @<letitgo>$@$<strong>{0123456789 ijkl IJKL !"$%&()*,-./:;?@[\\]`|~}
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//emlist{
タイプライタ体（通常）： 0123456789 ijkl IJKL !"$%&()*,-./:;?@[\]`|~
タイプライタ体（太字）： @<b>{0123456789 ijkl IJKL !"$%&()*,-./:;?@[\\]`|~}
ゴシック体　　（太字）： @<strong>{0123456789 ijkl IJKL !"$%&()*,-./:;?@[\\]`|~}
//}

//sampleoutputend




=== 目立たせないための「@<code>$@<nop>{@}<weak>{}$」

強調とは逆に、テキストを目立たせないための「@<code>$@<nop>{@}<weak>{}$」という命令も用意しました。
いわゆる「おまじない」のコードを目立たなくさせるときに使うといいでしょう。

次のはPHPのおまじないを目立たなくした例です。

//emlist[サンプル]{
@<letitgo>$//$list[][PHPサンプルコード]{
@<b>|@|@<b>|<weak>{|<?php@<b>|}|
function fib($n) {
    return $n <= 1 ? $n : fib($n-1) + fib($n-2);
}
@<b>|@|@<b>|<weak>{|?>@<b>|}|
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[][PHPサンプルコード]{
@<weak>{<?php}
function fib($n) {
    return $n <= 1 ? $n : fib($n-1) + fib($n-2);
}
@<weak>{?>}
//}

//sampleoutputend



次のはJavaのおまじないを目立たなくした例です。

//emlist[サンプル]{
@<letitgo>$//$list[][Javaサンプルコード]{
@<b>|@|@<b>|<weak>$|public class Example {@<b>|$|
    @<b>|@|@<b>|<weak>$|public static void main(String[] args) {@<b>|$|
        System.out.println("Hello!");
    @<b>|@|@<b>|<weak>$|}@<b>|$|
@<b>|@|@<b>|<weak>$|}@<b>|$|
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[][Javaサンプルコード]{
@<weak>$public class Example {$
    @<weak>$public static void main(String[] args) {$
        System.out.println("Hello!");
    @<weak>$}$
@<weak>$}$
//}

//sampleoutputend




=== 文字の大きさを変更する

Starterでは、文字の大きさを変更するインラインコマンドを用意しました（@<table>{ze5mn}）@<fn>{2muty}。
//footnote[2muty][「xsmall」や「xxsmall」という名前は、CSSの「x-small」や「xx-small」に由来します。]

//table[ze5mn][文字の大きさを変更するインラインコマンド]{
インラインコマンド	表示例
----------------------------------------
@<code>$@<nop>{@}<small>{...}$	@<small>{あいうabc123}
@<code>$@<nop>{@}<xsmall>{...}$	@<xsmall>{あいうabc123}
@<code>$@<nop>{@}<xxsmall>{...}$	@<xxsmall>{あいうabc123}
@<code>$@<nop>{@}<large>{...}$	@<large>{あいうabc123}
@<code>$@<nop>{@}<xlarge>{...}$	@<xlarge>{あいうabc123}
@<code>$@<nop>{@}<xxlarge>{...}$	@<xxlarge>{あいうabc123}
//}

また文字を太字ゴシック体にする「@<code>$@<nop>{@}<strong>{}$」の、文字を大きくする派生コマンドも用意しました（@<table>{0g6fo}）。

//table[0g6fo][文字を大きくする機能をもった「@<code>$@<nop>{@}<strong>{}$」]{
インラインコマンド	表示例
----------------------------------------
@<code>$@<nop>{@}<strong>{...}$	@<strong>{あいうabc123}
@<code>$@<nop>{@}<xtrong>{...}$	@<xstrong>{あいうabc123}
@<code>$@<nop>{@}<xxstrong>{...}$	@<xxstrong>{あいうabc123}
//}


==={subsec-nestable} 入れ子のインライン命令

Re:VIEWではインライン命令を入れ子にできませんが、Starterではできます。

たとえば次の例では、等幅フォントに変更する「@<code>$@<nop>{@}<tt>{...}$」の中にイタリック体にする「@<code>$@<nop>{@}<i>{...}$」が入っています。
つまりインライン命令@<fn>{c9r69}が入れ子になっています。

//footnote[c9r69][「インライン命令」とは、文章の途中に埋め込める「@<code>$@<nop>{@}<xxx>{...}$」形式の命令のことです。なお「@<code>$//xxx[\]{ ... //}$」形式の命令は「ブロック命令」といいます。]

//emlist[サンプル]{
Try @<letitgo>$@$<code>{git config user.name @<letitgo>$@$<i>{yourname}} at first.
//}



これをRe:VIEWでコンパイルすると、次のような表示になります。
入れ子のインライン命令を解釈できていないことが分かります。

//sampleoutputbegin[表示結果]

Try @<code>{git config user.name @<nop>{@}<i>{yourname}} at first.

//sampleoutputend



これに対し、Starterだと次のような表示になります。
入れ子のインライン命令をきちんと解釈できていますね。

//sampleoutputbegin[表示結果]

Try @<code>{git config user.name @<i>{yourname}} at first.

//sampleoutputend



なお「@<code>$@<nop>{@}<m>{...}$」と「@<code>$@<nop>{@}<raw>{...}$」と「@<code>$@<nop>{@}<embed>{...}$」は特別で、わざと入れ子に対応していません。
つまり他のインライン命令を受け取ってもただのテキストとして処理します。

またインライン命令を入れ子にしたくない場合は、内側のインライン命令の「@<code>{@}」を「@<code>$@<nop>{@}<nop>{@}$」@<fn>{pv0b2}のように書いてください。

//emlist[サンプル]{
次の例はインライン命令が入れ子になっている。

@<letitgo>$@$<code>{git config user.name @<letitgo>$@$<i>{yourname}}

次の例はインライン命令の入れ子を避けている。

@<letitgo>$@$<code>{git config user.name @<letitgo>$@$<nop>{@}<i>{yourname}}
//}

//sampleoutputbegin[表示結果]

次の例はインライン命令が入れ子になっている。

@<code>{git config user.name @<i>{yourname}}

次の例はインライン命令の入れ子を避けている。

@<code>{git config user.name @<nop>{@}<i>{yourname}}

//sampleoutputend



//footnote[pv0b2][「@<code>$@<nop>{@}<nop>{...}$」は引数を何も加工せず出力するためのインラインコマンドです。詳しくは@<secref>{yan75}を参照してください。]


==={subsec-qiqmo} 引数が2つ以上あるインライン命令での仕様追加

インライン命令を入れ子対応にしたことで、引数が2つ以上あるようなインライン命令の仕様を少し変更しました。

 * インライン命令「@<code>$@<nop>{@}<href>{@<i>{url}, @<i>{text}}$」は、URLでリンクを作成します。
 ** Re:VIEWでは、URLに「@<code>{,}」が含まれる場合は「@<code>{\,}」のようにエスケープする必要があります。しかしこの仕様は分かりにくいです。
 ** Starterでは、セパレータが「@<code>{, }」のように半角空白を含む場合は、URL中の「@<code>{,}」をエスケープしなくても済むよう仕様を変更しています。

 * インライン命令「@<code>$@<nop>{@}<ruby>{@<i>{text}, @<i>{yomi}}$」は、テキストによみがなをつけます。
 ** Re:VIEWではよみがなに「@<code>{,}」が含まれる場合、「@<code>{\,}」のようにエスケープする必要があります。しかしこの仕様は分かりにくいです。
 ** Starterではセパレータが「@<code>{, }」のように半角空白を含む場合は、よみがな中の「@<code>{,}」をエスケープしなくても済むよう仕様を変更しています。

//emlist[サンプル]{
Re:VIEWの場合はURL中の「,」にエスケープが必要

例：@<letitgo>$@$<href>{https://example.com/latlng/35.4\,135.5, 北緯35.4度・東経135.5度}

StarterではURL中の「,」にエスケープは不要
（引数のセパレータが「, 」の場合のみ。「,」ならエスケープ必要）

例：@<letitgo>$@$<href>{https://example.com/latlng/35.4,135.5, 北緯35.4度・東経135.5度}
//}

//sampleoutputbegin[表示結果]

Re:VIEWの場合はURL中の「,」にエスケープが必要

例：@<href>{https://example.com/latlng/35.4\,135.5, 北緯35.4度・東経135.5度}

StarterではURL中の「,」にエスケープは不要
（引数のセパレータが「, 」の場合のみ。「,」ならエスケープ必要）

例：@<href>{https://example.com/latlng/35.4,135.5, 北緯35.4度・東経135.5度}

//sampleoutputend



//note[インライン命令の入れ子対応と複数の引数]{
複数の引数を受け取るインライン命令の文法は、実は入れ子のインライン命令と相性がよくありません。
望ましいのは「@<code>$@<nop>{@}<href url="@<i>{url}">{@<i>{text}}$」や「@<code>$@<nop>{@}<ruby yomi="@<i>{yomi}">{@<i>{text}}$」のような文法なので、採用を現在検討中です。
//}


==={subsec-olist} 番号つきリストの機能強化

Re:VIEWでは番号つきリストを次のように書きます。

//emlist[サンプル]{
 1. XXX
 2. YYY
 3. ZZZ
//}

//sampleoutputbegin[表示結果]

 1. XXX
 2. YYY
 3. ZZZ

//sampleoutputend



この書き方には次の欠点があります。

 * 数字の番号はつきますが、「A.」や「a.」などは使えません。
 * また番号つきリストを入れ子にできません。

そこでStarterでは別の書き方を用意しました。

//emlist[サンプル]{
数字による番号つきリスト

 - 1. XXX
 - 2. YYY
 - 3. ZZZ

大文字による番号つきリスト

 - A. XXX
 - B. YYY
 - C. ZZZ

小文字による番号つきリスト

 - a. XXX
 - b. YYY
 - c. ZZZ
//}

//sampleoutputbegin[表示結果]

数字による番号つきリスト

 - 1. XXX
 - 2. YYY
 - 3. ZZZ

大文字による番号つきリスト

 - A. XXX
 - B. YYY
 - C. ZZZ

小文字による番号つきリスト

 - a. XXX
 - b. YYY
 - c. ZZZ

//sampleoutputend



「1.」や「A.」や「a.」のあとに必ず半角空白が必要です。
実は半角空白があれば、その前に書いた文字列がそのまま出力されます。
なので次のような書き方もできます。
箇条書きのように見えますが、「・」がついてないことに注意してください。

//emlist[サンプル]{
 - (A) 項目A
 - (B) 項目B
 - (C) 項目C

 - 甲: 山田太郎
 - 乙: 佐藤花子
//}

//sampleoutputbegin[表示結果]

 - (A) 項目A
 - (B) 項目B
 - (C) 項目C

 - 甲: 山田太郎
 - 乙: 佐藤花子

//sampleoutputend



また入れ子にできます。

//emlist[サンプル]{
 - (A) 作業A
 -- (A-1) 作業A-1
 -- (A-2) 作業A-2
//}

//sampleoutputbegin[表示結果]

 - (A) 作業A
 -- (A-1) 作業A-1
 -- (A-2) 作業A-2

//sampleoutputend



箇条書きとの混在もできます。

//emlist[サンプル]{
番号つきリストの中に箇条書き

 - A. XXX
 ** xxx
 ** xxx

箇条書きの中に番号つきリスト

 * XXXX
 -- a. xxx
 -- b. xxx
//}

//sampleoutputbegin[表示結果]

番号つきリストの中に箇条書き

 - A. XXX
 ** xxx
 ** xxx

箇条書きの中に番号つきリスト

 * XXXX
 -- a. xxx
 -- b. xxx

//sampleoutputend



なお数字や大文字や小文字の順番を補正するようなことはしません。
たとえば「1.」を連続して書けばそれがそのまま出力されます。

//emlist[サンプル]{
 - 1. XXX
 - 1. YYY
 - 1. ZZZ
//}

//sampleoutputbegin[表示結果]

 - 1. XXX
 - 1. YYY
 - 1. ZZZ

//sampleoutputend




==={subsec-ext-note} ノート

「@<code>$//note[$...@<code>$]{$ ... @<code>$//}$」で、付加情報や注意書きのブロックが書けます。
Re:VIEW標準と比べると、デザインを大きく変更していることと、段落の先頭は1文字分インデントされる点が違います。

//emlist[サンプル]{
@<letitgo>$//$note[■注意：印刷所の締切り日を確かめること]{
印刷所の締切りは、技術書典のようなイベントの本番当日よりずっと前です。
通常は約1週間前、また割引きを受けようと思ったら約2週間前が締切りです。
実際の締切り日は印刷所ごとに違うので、必ず確認しておきましょう。

また他の人に原稿のレビューを頼む場合は、さらに1〜2週間必要です。
これも忘れやすいので注意しましょう。
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//note[■注意：印刷所の締切り日を確かめること]{
印刷所の締切りは、技術書典のようなイベントの本番当日よりずっと前です。
通常は約1週間前、また割引きを受けようと思ったら約2週間前が締切りです。
実際の締切り日は印刷所ごとに違うので、必ず確認しておきましょう。

また他の人に原稿のレビューを頼む場合は、さらに1〜2週間必要です。
これも忘れやすいので注意しましょう。
//}

//sampleoutputend



実はRe:VIEWでは、ノートの中に箇条書きや他のブロック命令を含められません。これは技術同人誌や書籍の執筆において、大変困る欠点です。

なのでStarterではこれを解決し、ノートの中に箇条書きや他のブロック命令を含められるようにしました@<fn>{wosu0}@<fn>{mg7ep}。

//footnote[wosu0][以前はこれができなかったので、「@<code>$====[note\]$ ... @<code>$====[/note\]$」という別の記法が必要でした。今でもこの記法は有効ですが、もう使う必要はありません。]
//footnote[mg7ep][昔はノート中のプログラム（「@<code>$//emlist$」や「@<code>$//cmd$」）やターミナル（「@<code>$//terminal$」）がページをまたげないという制限がありましたが、現在はその制限はなくなりました。]

//emlist[サンプル]{
@<b>|//note[■ノートサンプル]{|

箇条書きを含める例@<letitgo>$@$<fn>{t71o9}。

 * エマ
 * レイ
 * ノーマン

@<letitgo>$//$footnote[t71o9][ノートの中に脚注を含めるサンプル。]

他のブロックを含める例。

@<letitgo>$//$emlist[RubyでHello]{
def hello(name)
  print("Hello, #{name}!\n")
end
@<letitgo>$//$}

@<letitgo>$//$cmd[UNIXでHello]{
$ echo Hello
Hello
@<letitgo>$//$}

@<b>|//}|
//}

//sampleoutputbegin[表示結果]

//note[■ノートサンプル]{

箇条書きを含める例@<fn>{t71o9}。

 * エマ
 * レイ
 * ノーマン

//footnote[t71o9][ノートの中に脚注を含めるサンプル。]

他のブロックを含める例。

//emlist[RubyでHello]{
def hello(name)
  print("Hello, #{name}!\n")
end
//}

//cmd[UNIXでHello]{
$ echo Hello
Hello
//}

//}

//sampleoutputend



なお「@<code>$//note$」機能はRe:VIEWの標準機能であり、Starterはそれを上書きしています。
実はRe:VIEWの標準のままだと、次のような表示になります。

//emlist[サンプル]{
@<letitgo>$//$note[印刷所の締切り日を確かめること]{
印刷所の締切りは、技術書典のようなイベントの本番当日よりずっと前です。
通常は約1週間前、また割引きを受けようと思ったら約2週間前が締切りです。
実際の締切り日は印刷所ごとに違うので、必ず確認しておきましょう。

また他の人に原稿のレビューを頼む場合は、さらに1〜2週間必要です。
これも忘れやすいので注意しましょう。
@<letitgo>$//$}
//}

//sampleoutputbegin[表示例（Re:VIEWのデフォルト）：]

//memo[印刷所の締切り日を確かめること]{
印刷所の締切りは、技術書典のようなイベントの本番当日よりずっと前です。
通常は約1週間前、また割引きを受けようと思ったら約2週間前が締切りです。
実際の締切り日は印刷所ごとに違うので、必ず確認しておきましょう。

また他の人に原稿のレビューを頼む場合は、さらに1〜2週間必要です。
これも忘れやすいので注意しましょう。
//}

//sampleoutputend



段落の先頭がインデントされてないことが分かります。
また、ノート（「@<code>$//note$」）なのになぜかキャプションが「■メモ：」になってる！
おかしいですよね。
詳しくは@<secref>{chap02-faq|subsec-faq-memo}を参照のこと。


=== ノートの参照

Starterでは、ノートにラベルをつけて他の箇所から参照できます。

//emlist[サンプル]{
@<letitgo>$//$note@<b>|[note-123]|[推しマンガ『鍵つきテラリウム』の紹介]{
『少女終末旅行』や『メイドインアビス』や『風の谷のナウシカ』（原作版）が好きな人は、お願いだから『鍵つきテラリウム』を読んで！
@<letitgo>$//$}

『鍵つきテラリウム』については@<b>|@<letitgo>$@$<noteref>{chap01-starter|note-123}|を参照してください。
//}

//sampleoutputbegin[表示結果]

//note[note-123][推しマンガ『鍵つきテラリウム』の紹介]{
『少女終末旅行』や『メイドインアビス』や『風の谷のナウシカ』（原作版）が好きな人は、お願いだから『鍵つきテラリウム』を読んで！
//}

『鍵つきテラリウム』については@<noteref>{chap01-starter|note-123}を参照してください。

//sampleoutputend



ここで「@<code>{chap01-starter}」は章IDであり、たとえばファイル名が「@<em>{chap01-starter.re}」なら拡張子を除いた「@<em>{chap01-starter}」が章IDです。
また同じ章のノートを参照するなら章IDを省略して「@<code>$@<nop>{@}<noteref>{note-123}$」のように書けます。

なお「@<code>$//note[タイトル]$」は「@<code>$//note[][タイトル]$」と同じだとみなされます。


=== プログラムコード用のコマンドを統一

Re:VIEWでは、プログラムコードを書くためのブロックコマンドが複数あります。

 : @<code>$//list[@<i>{ID}][@<i>{caption}][@<i>{lang}]$
	リスト番号あり、行番号なし
 : @<code>$//emlist[@<i>{caption}][@<i>{lang}]$
	リスト番号なし、行番号なし
 : @<code>$//listnum[@<i>{ID}][@<i>{caption}][@<i>{lang}]$
	リスト番号あり、行番号あり
 : @<code>$//emlistnum[@<i>{caption}][@<i>{lang}]$
	リスト番号なし、行番号あり

Starterでは、これらをすべて「@<code>$//list[][][]$」に統一しました。
それ以外のコマンドは、実質的に「@<code>$//list[][][]$」へのエイリアスとなります@<fn>{78vwj}。

//footnote[78vwj][「@<code>{//emlist}」や「@<code>{listnum}」が使えなくなったわけではありません。これらも引き続き使えますが、動作は「@<code>{//list}」を呼び出すだけになりました。]

 * 第1引数が空だと、「リストX.X:」のような番号がつきません。
   つまり「@<code>$//emlist$」と同じです。
 * 第3引数に「@<code>{lineno=on}」をつけると、行番号がつきます。
   つまり「@<code>$//listnum$」と同じです。
 * 第1引数を空にし、第3引数に「@<code>{lineno=on}」をつけると、リスト番号がつかず行番号がつきます。
   つまり「@<code>$//emlistnum$」と同じです。

//emlist[サンプル]{
@<b>|//list[4k2ny][リスト番号あり]|{
def fib(n)
  n <= 1 ? n : fib(n-1) + fib(n-2)
end
@<letitgo>$//$}

@<b>|//list[][リスト番号なし]|{
def fib(n)
  n <= 1 ? n : fib(n-1) + fib(n-2)
end
@<letitgo>$//$}

@<b>|//list[970bl][リスト番号あり、行番号あり][lineno=on]|{
def fib(n)
  n <= 1 ? n : fib(n-1) + fib(n-2)
end
@<letitgo>$//$}

@<b>|//list[][リスト番号なし、行番号あり][lineno=on]|{
def fib(n)
  n <= 1 ? n : fib(n-1) + fib(n-2)
end
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[4k2ny][リスト番号あり]{
def fib(n)
  n <= 1 ? n : fib(n-1) + fib(n-2)
end
//}

//list[][リスト番号なし]{
def fib(n)
  n <= 1 ? n : fib(n-1) + fib(n-2)
end
//}

//list[970bl][リスト番号あり、行番号あり][lineno=on]{
def fib(n)
  n <= 1 ? n : fib(n-1) + fib(n-2)
end
//}

//list[][リスト番号なし、行番号あり][lineno=on]{
def fib(n)
  n <= 1 ? n : fib(n-1) + fib(n-2)
end
//}

//sampleoutputend



リスト番号もキャプションも行番号もつけない場合は、すべての引数を省略して「@<code>$//list{ ... //}$」のように書けます。
この書き方はRe:VIEWではエラーになりますが、Starterではエラーになりません。

//emlist[サンプル]{
@<b>|//list|{
function fib(n) {
    return n <= 1 ? n : fib(n-1) + fib(n-2);
}
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list{
function fib(n) {
    return n <= 1 ? n : fib(n-1) + fib(n-2);
}
//}

//sampleoutputend



また「@<code>{//list}」の第3引数には、以下のオプションが指定できます。

 : @<code>$fold={on|off}$
	長い行を自動で折り返します（詳しくは後述）。
	デフォルトは@<code>{on}。
 : @<code>$foldmark={on|off}$
	折り返したことを表す、小さな記号をつけます。
	デフォルトは@<code>{on}。
 : @<code>$eolmark={on|off}$
	すべての行末に、行末であることを表す小さな記号をつけます。
	「@<code>$foldmark=on$」のかわりに使うことを想定していますが、両方を@<code>{on}にしても使えます。
	デフォルトは@<code>{off}。
 : @<code>$lineno={on|off|@<i>{integer}|@<i>{pattern}}$
	行番号をつけます。
	行番号は1から始まりますが、整数を指定するとそれが最初の行番号になります。
	またより複雑なパターンも指定できます（後述）。
	デフォルトは@<code>{off}。
 : @<code>$linenowidth=@<i>{integer}$
	行番号の桁数を指定します（詳しくは後述）。
	0だと自動計算します。
	値が0以上だと、行番号の分だけプログラムコードの表示幅が狭くなります。
	値がマイナスだと行番号はページの右余白に書かれるので、プログラムコードの表示幅が狭くなりません。
	デフォルトは@<code>{-1}。
 : @<code>$fontsize={small|x-small|xx-small|large|x-large|xx-large}$
	文字の大きさを小さく（または大きく）します。
	どうしてもプログラムコードを折返ししたくないときに使うといいでしょう。
 : @<code>$indentwidth=@<i>{integer}$
	インデント幅を指定します。
	たとえば「@<code>$indentwidth=4$」が指定されると、4文字幅のインデントを表すパイプ記号「@<code>$|$」がつきます。
	Pythonのようにブロックの構造をインデント幅で表す（ブロックの終わりを表す記号がない）ようなプログラミング言語の場合に使うといいでしょう。
	インデント幅を調整する機能ではないので注意してください。
 : @<code>$lang=@<i>{name}$
	プログラミング言語名を表します。
	デフォルトはなし。

いくつか補足事項があります。

 * 複数のオプションを指定するときは、「@<code>{,}」で区切ってください。
   たとえば「@<code>{//list[][][eolmark=on,lineno=on,linenowidth=3]}」のようにします。
 * オプションの名前だけを指定して値を省略すると、「@<code>{on}」を指定したとみなされます。
   たとえば「@<code>{lineno}」は「@<code>{lineno=on}」と同じです。
 * 「@<code>$lang=@<i>{name}$」を指定してもコードハイライトはできません。
   この制限は将来改善される予定ですが、時期は未定です。
 * 「@<code>$lang=@<i>{name}$」の場合は、省略形は「@<code>$lang$」ではなく「@<code>$@<i>{name}$」です@<fn>{auf8z}。
   またこの省略ができるのは、第3引数の最初のオプションに指定した場合だけです。
   つまり、「@<code>$ruby,lineno=1$」はOKだけど「@<code>$lineno=1,ruby$」はエラーになります。

//list[][]{
@<nop>{}これはOK
@<nop>{}//list[][][@<b>{ruby},lineno=1]{
@<nop>{}//}

@<nop>{}これはエラー
@<nop>{}//list[][][lineno=1,@<b>{ruby}]{
@<nop>{}//}

@<nop>{}これはOK
@<nop>{}//list[][][lineno=1,@<b>{lang=ruby}]{
@<nop>{}//}
//}

//footnote[auf8z][これはRe:VIEWとの互換性を保つために仕方なく決めた仕様なので、できれば「@<code>$lang={name}$」と省略せずに書いてください。この省略のせいでオプション名が間違っていても言語名とみなされてしまうので注意してください。]


=== ターミナル画面を表す「@<code>$//terminal$」ブロック

Starterでは、ターミナル画面用の新しいブロック命令「@<code>$//terminal{$ ... @<code>$//}$」を用意しました@<fn>{mthjy}。
これは「@<code>$//cmd{$ ... @<code>$//}$」とよく似ていますが、オプションの指定方法が「@<code>$//list{$ ... @<code>$//}$」と同じになっています。

//footnote[mthjy][@<code>$//terminal$命令の定義は@<code>$lib/hooks/monkeypatch.rb$で行っています。]

次の例を見てください。

 * 「@<code>$//cmd$」はオプション引数としてキャプションしか取れません。
   そのためリスト番号をつけられないし、行番号もつけられません。

//emlist[サンプル]{
@<b>|//cmd[キャプション]|{
$ echo foobar
foobar
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//cmd[キャプション]{
$ echo foobar
foobar
//}

//sampleoutputend



 * 「@<code>$//terminal$」はオプション引数が「@<code>$//list$」と同じです。
   そのためリスト番号をつけたり、行番号をつけることが簡単にできます。

//emlist[サンプル]{
@<b>|//terminal[id6789][キャプション][lineno=on]|{
$ echo foobar
foobar
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//terminal[id6789][キャプション][lineno=on]{
$ echo foobar
foobar
//}

//sampleoutputend



なおStarterでは、「@<code>{//cmd}」は実質的に「@<code>{//terminal}」を呼び出しているだけです。
なので上で説明したこと以外では、両者の機能は同じです。


=== プログラムコード中の長い行を自動的に折り返す

Starterでは、プログラムやターミナルの中の長い行を自動的に折り返します。

//emlist[サンプル]{
@<letitgo>$//$list[][長い行を含むプログラム例]{
data = <<HERE
123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_
HERE
@<letitgo>$//$}

@<letitgo>$//$terminal[][長い行を含む出力例]{
$ ruby foo/bar/baz/testprog.rb
foo/bar/baz/testprog.rb:11:in `func1': undefined local variable or method `aaabbbccc' for main:Object (NameError)
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[][長い行を含むプログラム例]{
data = <<HERE
123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_
HERE
//}

//terminal[][長い行を含む出力例]{
$ ruby foo/bar/baz/testprog.rb
foo/bar/baz/testprog.rb:11:in `func1': undefined local variable or method `aaabbbccc' for main:Object (NameError)
//}

//sampleoutputend



いくつか注意事項があります。

 * 折り返した行には、折り返したことを表す小さな記号がつきます。
   これをつけなくない場合は、「@<code>{//list}」や「@<code>{//terminal}」の第3引数に「@<code>{foldmark=off}」を指定してください。
 * 折り返すはずの箇所が日本語の場合、折り返しを表す記号が挿入されません@<fn>{wilge}。
   日本語の途中で折り返しをしたい場合は、手動で「@<code>$@<nop>{@}<foldhere>{}$」を挿入してください。
 * 右端にまだ文字が入るスペースがあるのに折り返しされている（ように見える）場合があります。
   この場合、プログラムやターミナルの表示幅をほんの少し広げると、右端まで文字で埋まるようになります。
   詳しくは@<secref>{chap02-faq|ikumq}を参照してください。
 * 折り返し機能によって何らかの問題が発生したら、「@<code>{//list}」や「@<code>{//terminal}」の第3引数に「@<code>{fold=off}」を指定して折り返し機能をオフにしてください。
   これは原因の切り分けに役立つでしょう。
 * 折り返し箇所で太字（@<code>$@<nop>{@}<b>{...}$）や取り消し線（@<code>$@<nop>{@}<del>{...}$）を使っていても@<fn>{kt429}、折り返しはされるし折り返し記号もつきます（@<list>{u4yj2}）。
   実は@<LaTeX>{}でこれを実現するのは簡単ではないのですが@<fn>{mpo3t}、頑張って実現しました。

//list[u4yj2][折り返し箇所が太字や取り消し線でも折り返しされる]{
@<b>{123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_}
@<del>{123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_}
//}

//footnote[wilge][英数字なら折り返し改行される位置にハイフンが入ります。このハイフンを強引に置き換えることで、折り返し記号を挿入しています。しかしpLaTeXでは日本語だとハイフンが入らないため、折り返し記号も挿入されません。これの解決は難しそうなので、別の方法を模索中。]
//footnote[mpo3t][通常は、取り消し線「@<code>$\sout{...}$」の中では折返しを実現する「@<code>$\seqsplit{...}$」が効かなくなります。Starterでは「@<code>$\sout{...}$」が使う内部コマンドを上書きして強引に実現しています。]
//footnote[kt429][XeLaTeXでは取り消し線がうまく動きません。現在調査中。]


//note[折り返し記号のかわりに行末記号]{

折り返し箇所が日本語だと折り返し記号がうまく挿入されません。
かといって手動で「@<code>$@<nop>{@}<foldhere>{}$」を挿入するのも面倒です。

そのような場合は、折り返し記号をオフにし、かわりに行末記号を入れることを検討してください。

次がその例です。折り返し記号はありませんが、行末記号があるので、行末記号がない箇所は折り返しされていることがわかります。

//emlist[サンプル]{
@<letitgo>$//$list[][]@<b>|[foldmark=off,eolmark=on]|{
def emergency()
  abort '深刻なエラーが発生しました。今すぐシステム管理者に連絡して、対処方法を仰いでください。'
end
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[][][foldmark=off,eolmark=on]{
def emergency()
  abort '深刻なエラーが発生しました。今すぐシステム管理者に連絡して、対処方法を仰いでください。'
end
//}

//sampleoutputend



//}


=== プログラムやターミナルの行番号を出力

Starterでは、プログラムやターミナルに行番号をつけられます。

//emlist[サンプル]{
@<letitgo>$//$list[][][@<b>|lineno=on|]{
function fib(n) {
  return n <= 1 ? n : fib(n-1) + fib(n-2);
}
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[][][lineno=on]{
function fib(n) {
  return n <= 1 ? n : fib(n-1) + fib(n-2);
}
//}

//sampleoutputend



正の整数を指定すると、最初の行番号になります。

//emlist[サンプル]{
@<letitgo>$//$list[][][@<b>|lineno=98|]{
function fib(n) {
  return n <= 1 ? n : fib(n-1) + fib(n-2);
}
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[][][lineno=98]{
function fib(n) {
  return n <= 1 ? n : fib(n-1) + fib(n-2);
}
//}

//sampleoutputend



行番号をつけるのにいちいち「@<code>{[lineno=1]}」と書くのが面倒な人のために、「@<code>{//list[][][lineno=1]}」を「@<code>{//list[][][1]}」と書けるようになりました。

//emlist[サンプル]{
@<letitgo>$//$list[][][@<b>|1|]{
function fib(n) {
  return n <= 1 ? n : fib(n-1) + fib(n-2);
}
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[][][1]{
function fib(n) {
  return n <= 1 ? n : fib(n-1) + fib(n-2);
}
//}

//sampleoutputend



行番号の桁数を指定すると、行番号が余白ではなく内側に表示されます。
その分、プログラムコードの表示幅が狭くなってしまいます。

//emlist[サンプル]{
@<letitgo>$//$list[][][lineno=98,@<b>|linenowidth=5|]{
function fib(n) {
  return n <= 1 ? n : fib(n-1) + fib(n-2);
}
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[][][lineno=98,linenowidth=5]{
function fib(n) {
  return n <= 1 ? n : fib(n-1) + fib(n-2);
}
//}

//sampleoutputend



行番号が灰色で表示されていることにも注目してください。
こうすることで、行番号とプログラムコードとの見分けがつきやすくなっています。

行番号の桁数に@<code>{0}を指定すると、表示幅を自動計算します。

//emlist[サンプル]{
@<letitgo>$//$list[][][lineno=98,@<b>|linenowidth=0|]{
function fib(n) {
  return n <= 1 ? n : fib(n-1) + fib(n-2);
}
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[][][lineno=98,linenowidth=0]{
function fib(n) {
  return n <= 1 ? n : fib(n-1) + fib(n-2);
}
//}

//sampleoutputend



長い行が折り返されたときは、折り返された行が左端からは始まらず、行番号の表示幅の分だけインデントされます。

//emlist[サンプル]{
@<letitgo>$//$list[][][lineno=1,linenowidth=2]{
data = <<HERE
123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_
HERE
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[][][lineno=1,linenowidth=2]{
data = <<HERE
123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_123456789_
HERE
//}

//sampleoutputend



行番号を表す、より複雑なパターンを指定できます。

 * 「@<code>{1-10}」なら、1行目から10行目まで
 * 「@<code>{1-10&15-18}」なら、1行目から10行目までと、1行空けて15行目から18行目まで
 * 「@<code>{1-10&15-}」なら、1行目から10行目までと、1行空けて15行目から最終行まで

サンプルを見ればどういうことか分かるでしょう。

//emlist[サンプル]{
@<letitgo>$//$list[][][@<b>|lineno=10&18-20&25-|]{
class Hello
  ...(省略)...
  def initialize(name)
    @name = name
  end
  ...(省略)...
  def hello
    print("Hello #{@name}\n")
  end

end
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[][][lineno=10&18-20&25-]{
class Hello
  ...(省略)...
  def initialize(name)
    @name = name
  end
  ...(省略)...
  def hello
    print("Hello #{@name}\n")
  end

end
//}

//sampleoutputend





=== ラベル指定なしでリスト番号を出力

リスト番号つきでソースコードを表示する場合、「@<code>$//list$」の第1引数にラベルを指定します。

//emlist[サンプル]{
@<letitgo>$//$list@<b>|[samplecode3]|[サンプル]{
puts "Hello"
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[samplecode3][サンプル]{
puts "Hello"
//}

//sampleoutputend



このラベルは、重複しないよう気をつけなければいけません。
リスト番号をあとから参照する場合は重複しないことが必要ですが、単にリスト番号をつけたい場合は重複しないラベルを選ぶのは面倒です。
特に、すべてのソースコードにリスト番号をつけようと思った場合はかなりの手間になります。

そこでStarterでは、「@<code>$//list[?]$」のように第1引数を「@<code>$?$」とするだけで、ラベルとしてランダムな文字列が割り当てられるようにしました@<fn>{w90w6}。
これにより、すべてのソースコードにリスト番号をつけるのが大幅に簡単になりました。
//footnote[w90w6][実装は@<em>{lib/hooks/monkeypatch.rb}の中で@<em>{ReVIEW::Book::Compilable#content()}を上書きして実現しています。]

//emlist[サンプル]{
@<letitgo>$//$list[@<b>|?|][サンプル]{
puts "Hello"
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[?][サンプル]{
puts "Hello"
//}

//sampleoutputend



この機能をサポートしているのは、次のブロック命令です。

 * @<code>$//list[?][@<i>{caption}]$ ... @<code>$//}$
 * @<code>$//listnum[?][@<i>{caption}]$ ... @<code>$//}$
 * @<code>$//terminal[?][@<i>{caption}]$ ... @<code>$//}$


=== キャプションなしでもリスト番号だけを出力

Re:VIEWでは、キャプションがないとリスト番号もつかない仕様です。
つまり「@<code>{//list[][]}」の第1引数を指定しても、第2引数が空ならリスト番号はつきません。
キャプションなしでリスト番号だけをつけたい場合は、第2引数に全角空白を入れます。

Starterではこの仕様を変更し、第1引数が指定してあれば第2引数が空（つまりキャプションが空）でもリスト番号をつけるようにしています。
こちらのほうが仕様として自然です。

//emlist[サンプル]{
@<b>|//list[test7][]|{
puts "Hello"
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list[test7][]{
puts "Hello"
//}

//sampleoutputend




=== プログラムのキャプション直後の改ページを抑制

Re:VIEWでは、プログラムやターミナルのキャプション（説明）直後に改ページされてしまうことがあります（@<img>{caption_pagebreak}）。
もしこうなると、キャプションが何を説明しているのか分かりにくくなります。
//image[caption_pagebreak][キャプションの直後で改ページされた例][scale=0.8]

Starterではこれを改善し、キャプションの直後では改ページを起こさないようにしました@<fn>{6yw2g}。
かわりにキャプションの直前で改ページされます。
//footnote[6yw2g][これは@<LaTeX>{}の@<em>{needspace.sty}で実現しています。]

ただし同じページに脚注が複数行あると、判定を間違えてキャプション直後に改ページされることがあります。
これは現在の制限事項です。
経験則として、キャプションの前の文章を増やすとなぜか治ることが多いです。


=== コラム内の脚注

Re:VIEWでは、コラムの中に書いた脚注が消えることがあります。
たとえば次のように書いた場合は、脚注が消えます。

//emlist[コラム内の脚注が消えるサンプル]{
==[column] サンプル
本文本文@<nop>{@}<fn>{xxx1}本文本文。

@<nop>{//}footnote[xxx1][脚注脚注脚注。]
//}

こうではなく、次のようにコラムを明示的に閉じてから脚注を書くと、消えずに表示されます。

//emlist[コラム内の脚注が消えないサンプル]{
==[column] サンプル
本文本文@<nop>{@}<fn>{xxx1}本文本文。

@<b>$==[/column]$

@<nop>{//}footnote[xxx1][脚注脚注脚注。]
//}

Starterではこの問題に対処するために、前者のように書かれた場合でも、後者のように自動変換します。

変換はスクリプト@<em>{lib/hooks/beforetexcompile.rb}が行います。
設定ファイルである@<em>{config.yml}に「@<code>$hook_beforetexcompile: [lib/hooks/beforetexcompile.rb]$」という設定を追加しているため、@<LaTeX>{}コマンドでコンパイルされる前にこのスクリプトが実行されるようになっています。


=== 右寄せ、左寄せ、センタリング

Starterでは、右寄せや左寄せやセンタリングをする機能を追加しました。

//emlist[サンプル]{
@<letitgo>$//$textright{
右寄せのサンプル
@<letitgo>$//$}
@<letitgo>$//$textleft{
左寄せのサンプル
@<letitgo>$//$}
@<letitgo>$//$textcenter{
センタリングのサンプル
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//textright{
右寄せのサンプル
//}
//textleft{
左寄せのサンプル
//}
//textcenter{
センタリングのサンプル
//}

//sampleoutputend



しかし、実はRe:VIEWにも右寄せとセンタリングの機能があることが判明しました。
今後はこちらを使うのがいいでしょう@<fn>{b9jz8}。
//footnote[b9jz8][ただし@<href>{https://github.com/kmuto/review/blob/master/doc/format.ja.md}には載ってないので、undocumentedな機能です。将来は変更されるかもしれません。]

//emlist[サンプル]{
@<letitgo>$//$flushright{
右寄せのサンプル
@<letitgo>$//$}
@<letitgo>$//$centering{
センタリングのサンプル
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//flushright{
右寄せのサンプル
//}
//centering{
センタリングのサンプル
//}

//sampleoutputend




=== 章の概要

Starterでは、章(Chapter)の概要を表す「@<code>$//abstract{$ ... @<code>$//}$」を用意しています。
#@#これは、@<LaTeX>{}の「@<em>$abstract$」環境と同等です。

//emlist[サンプル]{
@<letitgo>$//$abstract{
この章では、XXXのXXXという機能について説明します。
この機能を理解することで、あとの章が理解できるようになります。
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//abstract{
この章では、XXXのXXXという機能について説明します。
この機能を理解することで、あとの章が理解できるようになります。
//}

//sampleoutputend



本文と違う見た目にするために、デフォルトでは左右に全角2.5文字文の余白を追加し、かつゴシック体で表示します。
デザインを変更する場合は、@<em>{sty/starter.sty}で「@<code>$\newenvironment{starterabstract}$」を探し、変更してください。

なおこれとよく似た機能として、Re:VIEWには導入文（リード文）を表す「@<code>$//lead{$ ... @<code>$//}$」が標準で用意されています。
これは主に、詩や物語や聖書からの引用を表すのに使うようです（海外の本にはよくありますよね）。
そのため、「@<code>$//lead$」は@<LaTeX>{}での引用を表す「@<code>$quotation$」環境に変換されます。

//emlist[サンプル]{
@<letitgo>$//$lead{
土に根を下ろし　風と共に生きよう

種と共に冬を越え　鳥と共に春を歌おう
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//lead{
土に根を下ろし　風と共に生きよう

種と共に冬を越え　鳥と共に春を歌おう
//}

//sampleoutputend




=== 図が次のページに送られるときにスペースを空けない

Re:VIEWのデフォルトでは、図を入れるときに現在位置に入りきれない場合は、次のページに送られます。
それは仕方ないのですが、このとき現在位置に大きな空きができてしまいます（@<img>{figure_heretop}の上）。

//image[figure_heretop][図が次のページに送られると、そこに大きな空きができてしまう][scale=0.8]

これに対する解決策として、Starterでは空いたスペースに後続のテキストを流し込む選択肢を用意しています（@<img>{figure_heretop}の下）。

そのためには、Starterのプロジェクト作成ページに「画像が現在位置に入りきらず次のページに回されるとき、大きなスペースを空けない （かわりに後続のテキストを流し込む）」というチェックボックスがあるので、これを選んでください。
または、@<em>{config-starter.yml}で「@<code>{image_position:}」というオプションに「@<code>{h}」を指定してください。

またStarterでは「@<code>$//image$」コマンドを拡張し、図の挿入位置が指定できるようになっています@<fn>{u5dnl}。
これを指定することで、空いたスペースに後続のテキストを流し込むかどうかを、画像ごとに制御できます。
//footnote[u5dnl][実装方法は@<em>{lib/hooks/monkeypatch.rb}を見てください。]

 * 「@<code>$//image[][][@<strong>{pos=H}]$」なら後続のテキストを流し込まない@<br>{}
   （つまり画像を現在位置に強制的に配置する）
 * 「@<code>$//image[][][@<strong>{pos=h}]$」なら後続のテキストを流し込む@<br>{}
   （つまり画像が現在位置に入りきらなければ次のページの先頭に配置する）

画像の倍率も指定する場合は、「@<code>$//image[][][scale=0.5,pos=H]$」のように指定してください。


//note[ページ下部にも画像を配置する]{

「@<code>$pos=H$」や「@<code>$pos=h$」のどちらを選んでも、入りきらない画像は次ページに送られます。
そのため、どうしても画像はページ上部に配置されることが多くなり、逆にページ下部には配置されにくくなります。

このバランスの悪さが気になる場合は、（小さい画像を除いて大きめの画像に）「@<code>$pos=bt$」を指定してみてください。
ここで「@<code>$b$」はボトム(bottom)、「@<code>$t$」はトップ(top)を表します。
つまり、まずページ下部に配置を試み、入らないなら次ページ上部に配置します。
これで、全体的に図がページの上部と下部の両方に配置されるはずです。

//}

//note[「次の図」や「以下の図」という表現を止める]{

すでに説明したように、画像の配置場所として「@<code>$pos=H$」@<strong>$以外$を指定した場合は、後続のテキストが現在位置に流し込まれます。
そのため、文章中で「次の図は〜」とか「以下の図では〜」と書いていると、図が次ページに配置された場合、読者が混乱します。

このような事態を避けるために、「次の図は〜」や「以下の図では〜」という表現を止めて、「図1.1では〜」のように番号で参照するようにしましょう。
面倒でしょうが、仕方がありません。慣れてください。

//}


=== 図のまわりを線で囲む

Starterでは、図のまわりをグレーの線で囲むことができます。
そのためには「@<code>{//image}」の第3引数に「@<code>{border=on}」を指定します。

//emlist[サンプル]{
@<letitgo>$//$image[tw-icon][デフォルトの表示][scale=0.5,pos=H]

@<letitgo>$//$image[tw-icon][まわりを線で囲む][scale=0.5,pos=H,@<b>|border=on|]
//}

//sampleoutputbegin[表示結果]

//image[tw-icon][デフォルトの表示][scale=0.5,pos=H]

//image[tw-icon][まわりを線で囲む][scale=0.5,pos=H,border=on]

//sampleoutputend




==={yan75} 何もしない命令「@<code>$@<nop>{@}<nop>{...}$」

Re:VIEWでは、「@<code>$//list{ ... //}$」や「@<code>$//emlist{ ... //}$」のようなブロック命令の中で、「@<code>$@<nop>{@}<b>{...}$」などのインライン命令が利用できます。
そのため、「@<code>$//list{ ... //}$」の中で「@<code>$@<nop>{@}<b>{...}$」そのものを表示させるには、次のように「@<code>$@$」だけを「@<code>$@<nop>{@}<code>{...}$」で囲う（つまり「@<code>{@}」と「@<code>$<b>{}$」とを分離する）というトリックが必要です。

//emlist[サンプル]{
@<letitgo>$//$list{
  @<letitgo>$@$<b>{ABC}
  @<letitgo>$@$<code>{@}<b>{ABC}   ← 「@」と「<b>{}」とを分離する
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list{
  @<b>{ABC}
  @<code>{@}<b>{ABC}   ← 「@」と「<b>{}」とを分離する
//}

//sampleoutputend



この方法はうまく動作しますが、そもそもソースコードを表示するための「@<code>$//emlist{$ ... @<code>$//}$」の中で「@<code>$@<nop>{@}<code>{...}$」を使うのもおかしな話です。

そこでStarterでは、何もしないインライン命令「@<code>$@<nop>{@}<nop>{...}$」を用意しました（「nop」は「No Operation」の略です）。
これを使うと、引数を何も加工せず表示します。

これを使って「@<code>$//list{ ... //}$」の中で「@<code>$@<nop>{@}<b>{...}$」そのものを表示させるには次のようにします。

//emlist[サンプル]{
@<letitgo>$//$list{
  @<letitgo>$@$<nop>{@}           ← 「@」をそのまま表示する
  @<letitgo>$@$<nop>{@}<b>{ABC}   ← 「@」と「<b>{}」とを分離する
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//list{
  @<nop>{@}           ← 「@」をそのまま表示する
  @<nop>{@}<b>{ABC}   ← 「@」と「<b>{}」とを分離する
//}

//sampleoutputend



ただし、「@<code>$@<nop>{...}$」の中には他のインライン命令を入れないでください。
入れるとエラーになります。

//emlist[サンプル]{
@<letitgo>$//$emlist{
  @<letitgo>$@$<nop>$@<letitgo>$@$<b>{ABC}$   ← 他のインライン命令が入っているのでエラー
@<letitgo>$//$}
//}



なお「@<code>$@<nop>{@}<nop>{}$」はもともと「@<code>$@<nop>{@}<letitgo>{}$」という名前でしたが、長すぎるという意見があったので「@<code>$@<nop>{@}<nop>{}$」になりました。後方互換性のため、「@<code>$@<nop>{@}<letitgo>{}$」も使えます。


=== 章や項を参照する「@<code>$@<nop>{@}<secref>{}$」

Re:VIEWでは、「@<code>$@<nop>{@}<hd>{}$」を使って節(Section)や項(Subsection)を参照できます。
しかしこの機能には問題点があります。

 * Re:VIEWのデフォルト設定@<fn>{26mo5}では、章(Chapter)と節(Section)には番号がつくけど、項(Subsection)には番号がつかない。
 * そのため、「@<code>$@<nop>{@}<hd>{}$」で項(Subsection)を参照すると、番号がなくて項タイトルだけになるので文章がとても不自然になる。
//footnote[26mo5][Re:VIEWのデフォルトでは@<em>{config.yml}で「@<code>$secnolevel: 2$」と設定されています。これが3以上でないと、項(Subsection)に番号がつきません。]

サンプルを使って説明しましょう。
たとえば次のような原稿ファイルがあったとします。

//emlist[ファイル：chap-pattern.re]{
@<letitgo>$=$ デザインパターンの紹介

@<letitgo>$==${sec-visitor} Visitorパターン

@<letitgo>$===${subsec-motivation} 動機

@<letitgo>$===${subsec-structure} 構造

@<letitgo>$===${subsec-impl} 実装例

//}

文章の構造は次のようになっていますね。

 * 「デザインパターンの紹介」は章(Chapter)
 ** 「Visitorパターン」は節(Section)
 *** 「動機」と「構造」と「実装例」は項(Subsection)

さてRe:VIEWのデフォルト設定のままだと、次のように章と節には番号がつくけど、項には番号がつきません。

//sampleoutputbegin[表示結果]

//blankline
//embed[latex]{
\par\noindent
{\LARGE\headfont 第1章　デザインパターンの紹介}
\bigskip
\par\noindent
{\Large\headfont 1.1　Visitorパターン}
\bigskip
\par\noindent
{\large\headfont 動機}
\bigskip
\par\noindent
{\large\headfont  構造}
\bigskip
\par\noindent
{\large\headfont  実装例}
\par
//}
//blankline

//sampleoutputend



このことを踏まえたうえで、節や項を「@<code>$@<nop>{@}<hd>{}$」で参照するとどう表示されるでしょうか。

 * 節(Section)には番号がついているので、たとえば「@<code>$@<nop>{@}<hd>{sec-visitor}$」のように節を参照すると、「1.1 Visitorパターン」のように表示されます。
   これだと番号がついているので、読者は節を探しやすいです。
 * しかし項(Subsection)には番号がついていないので、たとえば「@<code>$@<nop>{@}<hd>{subsec-motivation}$」や「@<code>$@<nop>{@}<hd>{subsec-structure}$」のように項を参照すると、「動機」や「構造」とだけ表示されてしまいます。
   これだと番号がついていないので、読者は項を探せないでしょう。

//emlist[サンプル（最初の1つは節を参照、残り3つは項を参照）]{
 * @<nop>{@}<hd>{sec-visitor}
 * @<nop>{@}<hd>{subsec-motivation}
 * @<nop>{@}<hd>{subsec-structure}
 * @<nop>{@}<hd>{subsec-impl}
//}

//sampleoutputbegin[表示結果]


 * 「1.1 Visitorパターン」
 * 「動機」
 * 「構造」
 * 「実装例」


//sampleoutputend



問題点をもう一度整理しましょう。

 * Re:VIEWのデフォルト設定では、項(Subsection)に番号がつかない。
 * そのため、「@<code>$@<nop>{@}<hd>{}$」で項を参照するとタイトルだけになってしまい、番号がつかないので読者が項を探せない。

この問題に対し、Starterでは「@<code>$@<nop>{@}<secref>{}$」という新しい命令を用意しました。
この新命令には次のような利点があります。

 * 番号のついていない項でも、親となる節を使うことで探しやすい表示をしてくれる。
 * その項のページ番号がつくので、該当ページに直接アクセスできる。

次のサンプルを見れば、「@<code>$@$@<code>$<hd>{}$」との違いがすぐに分かるでしょう。

//emlist[サンプル（最初の1つは節を参照、残り3つは項を参照）]{
 * @<b>$@<nop>{@}<secref>${sec-visitor}
 * @<b>$@<nop>{@}<secref>${subsec-motivation}
 * @<b>$@<nop>{@}<secref>${subsec-structure}
 * @<b>$@<nop>{@}<secref>${subsec-impl}
//}

//sampleoutputbegin[表示結果]


 * 「1.1 Visitorパターン」(p.1)
 * 「1.1 Visitorパターン」内の「動機」(p.1)
 * 「1.1 Visitorパターン」内の「構造」(p.1)
 * 「1.1 Visitorパターン」内の「実装例」(p.1)


//sampleoutputend



これを見ると、番号がついていない項の前に番号がついている節が置かれていること、またページ番号がついていることが分かります。
どちらも@<code>$@<nop>{@}<hd>{}$にはない特徴であり、@<code>$@<nop>{@}<hd>{}$で参照するより節や項が探しやすくなります。

その他の注意事項です。

 * 「@<code>$@<nop>{@}<secref>{}$」は、節でも項でも、あるいは目(Subsubsection)でも参照できます。
   今まで「@<code>$@<nop>{@}<hd>{}$」を使っていた箇所はすべて「@<code>$@<nop>{@}<secref>{}$」で置き換えられます。
   ただし、章(Chapter)は参照できないので、その場合は「@<code>$@<nop>{@}<chapref>{}$」を使ってください。
 * 項にも番号をつけるよう設定している場合は、「@<code>$@<nop>{@}<secref>{}$」の表示結果は「@<code>$@<nop>{@}<hd>{}$」にページ番号をつけたものと同じです。
 * 他の章(Chapter)の節や項を参照する場合は、たとえば@<br>{}
  「@<code>$@<nop>{@}<secref>{@<i>{chapter-id}|sec-visitor}$」や@<br>{}
  「@<code>$@<nop>{@}<secref>{@<i>{chapter-id}|subsec-impl}$」のように書いてください。わざわざ@<br>{}
  「@<code>$@<nop>{@}<secref>{@<i>{chapter-id}|sec-visitor|subsec-impl}$」のように書く必要はありません。@<fn>{yacaz}

//footnote[yacaz][ここで「@<code>{@<i>{chapter-id}}」は章のIDを表します。たとえばファイルが「@<em>{foobar.rb}」なら拡張子を取り除いた「@<em>{foobar}」が章IDです。]

なおこの機能は、@<em>{config-starter.yml}の「@<code>{secref_parenttitle: @<i>{bool}}」で変更できます。
この値が@<code>{true}なら親となる節のタイトルを付け、@<code>{false}なら付けません。


=== 「@<code>$@<nop>{@}<chapref>{}$」や「@<code>$@<nop>{@}<hd>{}$」をリンクに

Starterでは、「@<code>$@<nop>{@}<chapref>{}$」や「@<code>$<nop>{@}<hd>{}$」がリンクになるように設定しています。
そのために次のような設定をしています。

 * @<em>{config.yml}に「@<code>$chapterlink: true$」という設定を追加（最終行）。
 * @<em>{sty/starter.sty}で「@<em>{\reviewsecref}」を再定義し、「@<em>{\hyperref}」を使うように変更。

//emlist[リスト：\reviewsecrefを再定義]{
\renewcommand{\reviewsecref}[2]{%
  \hyperref[#2]{#1}(p.\pageref{#2})%     % 節や項のタイトルがリンク
  %{#1}(\hyperref[#2]{p.\pageref{#2}})%  % ページ番号がリンク
}
//}

これらはRe:VIEWに実装済みの機能であり、Starterはそれらを有効化しただけです。
しかしこれらの機能はRe:VIEWのドキュメントには書かれていないので、もしかしたら将来的に変更されるかもしれません。

またStarterの追加機能である「@<em>{@}@<em>$<secref>{}$」でも、リンクが作成されます。


=== 画像とテキストを並べて表示する

Starterでは、画像とテキストを並べて表示するためのブロックコマンド「@<code>{//sideimage}」を用意しました。
著者紹介においてTwitterアイコンとともに使うといいでしょう。

//emlist[サンプル]{
@<b>|//sideimage[tw-icon][20mm][side=L,sep=7mm,border=on]|{
@<letitgo>$//$noindent
@<letitgo>$@$<strong>{カウプラン機関極東支部}

 * @<letitgo>$@$<href>{https://twitter.com/_kauplan/, @_kauplan}
 * @<letitgo>$@$<href>{https://kauplan.org/}
 * 技術書典7新刊「わかりみSQL」出します！
 * 「@<letitgo>$@$<href>{http://worldtrigger.info/, ワールド・トリガー}」連載再開おめ！

@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//sideimage[tw-icon][20mm][side=L,sep=7mm,border=on]{
//noindent
@<strong>{カウプラン機関極東支部}

 * @<href>{https://twitter.com/_kauplan/, @_kauplan}
 * @<href>{https://kauplan.org/}
 * 技術書典7新刊「わかりみSQL」出します！
 * 「@<href>{http://worldtrigger.info/, ワールド・トリガー}」連載再開おめ！

//}

//sampleoutputend



使い方は「@<code>$//sideimage[@<i>{画像ファイル}][@<i>{画像表示幅}][@<i>{オプション}]{$ ... @<code>$//}$」です。

 * 画像ファイルは「@<code>{//image}」と同じように指定します。
 * 画像表示幅は「@<code>{30mm}」「@<code>{3.0cm}」「@<code>{1zw}」「@<code>{10%}」などのように指定します。
   使用できる単位はこの4つであり、「@<code>{1zw}」は全角文字1文字分の幅、「@<code>{10%}」は本文幅の10%になります。
   なお「@<code>{//image}」と違い、単位がない「@<code>{0.1}」が10%を表すという仕様ではなく、エラーになります。
 * オプションはたとえば「@<code>{side=L,sep=7mm,boxwidth=40mm,border=on}」のように指定します。
 ** 「@<code>{side=L}」で画像が左側、「@<code>{side=R}」で画像が右側にきます。
    デフォルトは「@<code>{side=L}」。
 ** 「@<code>{sep=7mm}」は、画像と本文の間のセパレータとなる余白幅です。
    デフォルトはなし。
 ** 「@<code>{boxwidth=40mm}」は、画像を表示する領域の幅です。
    画像表示幅より広い長さを指定してください。
    デフォルトは画像表示幅と同じです。
 ** 「@<code>{border=on}」は、画像を灰色の線で囲みます。
    デフォルトはoff。

なお「@<code>{//sideimage}」は内部で@<LaTeX>{}のminipage環境を使っているため、次のような制限があります。

 * 途中で改ページされません。
 * 画像下へのテキストの回り込みはできません。
 * 脚注が使えません。

こういった制限があることを分かったうえで使ってください。


=== ターミナルでのカーソル

ターミナルでのカーソルを表す機能を用意しました。

//emlist[サンプル]{
@<letitgo>$//$terminal{
function fib(n) {
  return n <= 1 ? n : @<letitgo>$@$<cursor>{f}ib(n-1) : fib(n-2);
}
~
~
"fib.js" 3L, 74C written
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//terminal{
function fib(n) {
  return n <= 1 ? n : @<cursor>{f}ib(n-1) : fib(n-2);
}
~
~
"fib.js" 3L, 74C written
//}

//sampleoutputend



上の例では、2行目の真ん中の「@<code>{f}」にカーソルがあることを表しています。


=== その他

 * ブロック命令「@<code>$//clearpage$」で改ページします。
   また過去との互換性のために、インライン命令「@<code>$@<nop>{@}<clearpage>{}$」も使えます。
 * 「@<code>$@<nop>{@}<hearts>{}$」とすると、「@<hearts>{}」と表示されます。
 * 「@<code>$@<nop>{@}<TeX>{}$」とすると、「@<TeX>{}」と表示されます。
 * 「@<code>$@<nop>{@}<LaTeX>{}$」とすると、「@<LaTeX>{}」と表示されます。
 * 「@<code>$@<nop>{@}<ruby>{小鳥遊, たかなし}$」とすると、「@<ruby>{小鳥遊, たかなし}」と表示されます。
 * 「@<code>$@<nop>{@}<bou>{傍点}$」とすると、「@<bou>{傍点}」と表示されます。



=={sec-design} レイアウトやデザインに関する変更や拡張


=== Starterの設定ファイル「@<em>{config-starter.yml}」

Starterでは、「@<em>{config-starter.yml}」という設定ファイルを新たに用意しました。
この設定ファイルを編集することで、プロジェクトをダウンロードしたあとでもレイアウトやデザインを簡単に変更できます。

たとえば以下のことが変更できます。

 * PDFのターゲット（印刷用か、ダウンロード用か）
 * 章や節や項のデザイン
 * プログラムやターミナルの表示で使う等幅フォント
 * ドラフトモード（画像を枠線だけで表示する）

残念ながら、プロジェクト作成時にGUIで設定できたことがすべて「@<em>{config-starter.yml}」でできるわけではありません。しかしなるべく多くのことがこの設定ファイルで変更できるようにするつもりです。


=== フォントサイズの変更に対応

Re:VIEW 2.5は、標準では本文のフォントサイズを9ptや8ptに指定しても、効いてくれません（まじかー！）@<fn>{8tfuo}。
ウソだと思うかも知れませんが、実際に苦しんだ人の証言@<fn>{g6rdf}があるのでご覧ください（@<img>{slide2}）。
先人の苦労が偲ばれます。
//footnote[8tfuo][Re:VIEW 3からはできるようになりました。]
//footnote[g6rdf][@<href>{https://www.slideshare.net/KazutoshiKashimoto/nagoya0927-release}のp.21とp.22。]

//image[slide2][フォントやページサイズを変更できなかった人の証言]

この問題は、「@<em>{geometry.sty}」というスタイルファイルをオプションなしで読み込んでいることが原因です@<fn>{ly58b}。
Starterではこれを読み込まないように修正している@<fn>{z4ccg}ため、フォントサイズを9ptや8ptに指定すればそのサイズになります。
//footnote[ly58b][簡単に書いてますけど、原因が@<em>{geometry.sty}であることを突き止めるのには大変な時間がかかり、正月休みを潰してしまいました。コノウラミハラサデオクベキカ！]
//footnote[z4ccg][修正箇所は、@<em>{layouts/layout.tex.erb}の50行目あたりです。]


=== A5サイズの指定に対応
Re:VIEW 2.5は、標準ではA5サイズの指定が効いてくれません（まじかー！）。
ウソだと思うかも知れませんが、実際にトラブルに陥った人の証言があります@<fn>{m10ye}。
//footnote[m10ye][@<href>{https://blog.vtryo.me/entry/submit-of-journey}]

//quote{
　@<br>{}
@<strong>{トラブル発生！！}@<br>{}@<br>{}
原稿データチェック、表紙チェック、ともに問題なく終わったかにみえた午後1時。@<br>{}
何かに気づいたお姉さんの声音が変わりました。@<br>{}
「すみません、PDFサイズ……B5になってます」@<br>{}
「えっ……」@<br>{}
めのまえがまっくらになった。@<br>{}
セイチョウ・ジャーニーはA5で制作しているはずなのに、B5サイズに？？@<br>{}
わからない！どうして！だって何度も確認したはずだ！！@<br>{}
と度重なる徹夜で脳死寸前の僕はパニック状態になりました。@<br>{}
//}

入稿で明らかになるトラブル！怖いですねー。
こういう予期せぬトラブルがあるので、締切りギリギリまで作業するのは止めて、余裕をもって入稿しましょう。

さて、A5にならない問題は2種類あります。

 * 本文の大きさがA5サイズにならない。
 * 本文の大きさはA5なのにPDFがA5サイズにならない。

前者は、「@<em>{geometry.sty}」が原因です。
すでに説明したように、Starterでは「@<em>{geometry.sty}」を読み込まないようにしているため、この問題は起こりません。

後者は、上で紹介したトラブルですね。
これは@<em>{jsbook.sty}のオプションに「@<em>{papersize}」が指定されてないせいです。
Starterではこのオプションを指定しているので、A5やB5の指定どおりのPDFが生成されます。

詳しくは、@<em>{config.yml}の「@<code>$texdocumentclass:$」を参照してください。

//list[][config.yml][fontsize=small]{
texdocumentclass: ["jsbook",
    #"uplatex,papersize,twoside,b5j,10pt,openright" # B5 10pt 右起こし
    #"uplatex,papersize,twoside,b5j,10pt,openany"   # B5 10pt 両起こし
    #"uplatex,papersize,twoside,a5j,9pt,openright"  # A5  9pt 右起こし
    #"uplatex,papersize,twoside,a5j,9pt,openany"    # A5  9pt 両起こし
    #"uplatex,papersize,oneside,a5j,10pt,openany"   # A5 10pt 両起こし
    "uplatex,papersize,twoside,a5j,9pt,openright"
]
//}


=== 本文の幅を全角40文字より長くできる

@<LaTeX>{}の@<em>{jsbook.cls}ファイルを使うと、デフォルトでは本文の幅の最大値が40文字までに制限されています。
これは、1行が全角40文字より長いと読みづらくなるからという理由だそうです@<fn>{b1flb}。
//footnote[b1flb][@<href>{https://oku.edu.mie-u.ac.jp/~okumura/jsclasses/}に、@<em>{jsbook.cls}の作者である奥村先生が『書籍では1行の長さが全角40文字を超えないようにしています。』と解説しています。]

そのため、B5サイズだとページ左右の余白が広めになってしまいます。
ページ数を抑えて印刷代を下げたい人にとって、この余白はコストを増加させる要因です。

Starterでは@<em>{sty/mytextsize.sty}で本文幅を再設定することで、本文の幅を40文字より広くできます。
B5サイズでフォントが10ptだと、1行あたり全角42〜45文字がいいでしょう。

ただしA5サイズ（フォント9pt）では、1行あたり40文字を超えるのはやめたほうがいいです。
参考までに市販の技術書だと、A5サイズで1行あたり全角39文字にすることが多いようです。


==={jwevu} 奇数ページと偶数ページで左右の余白を変える

ページ数を抑えて印刷費を減らすために、余白をギリギリまで切り詰める人がいます。
しかしこれは読みやすさを大きく損なうので、止めたほうがいいです。
本文の幅を広げる場合でも、左右の余白はちゃんと取りましょう（@<img>{margin_book}）。

//image[margin_book][奇数ページと偶数ページで左右の余白を変える][scale=0.7]

 * 本を開いたときの中央（「ノド」という）の余白、つまり左ページの右余白と右ページの左余白は、最低でも2cmは確保しましょう。
   そうしないと、ノド近くの文章がとても読みづらくなります。
 * 本を開いたときの外側（「小口」という）の余白、つまり左ページの左余白と右ページの右余白は、最低でも1cmは確保しましょう。
   そうしないと、ページをめくるときに指が本文にかかってしまい、読みにくいです。

Starterでは1行あたりの文字数を増やしても読みやすさを保つために、綴じしろの余白は保ったままて外側の余白を減らしています。
そのため、@<img>{margin_book}のように左右の余白幅が異なることがあります。
これは意図的なことであり、不具合ではありません。

詳しくは@<em>{sty/mytextsize.sty}を見てください。


=== ページ上部の余白を減らし、その分を本文の高さに追加

@<LaTeX>{}の標準のデザインでは、ページ上部の余白が大きめになっています。
ページ数を少しでも減らして印刷代を抑えたい場合は、この余白がとても気になります。

Starterではこの余白を約1cm減らし@<fn>{g2ohz}、その分を本文の高さに追加しています。
詳しくは@<em>{sty/mytextsize.sty}を見てください。
//footnote[g2ohz][実はjsbook.clsでは「@<em>{1cm}」は1cmより少し大きく扱われ、厳密に1cmを指定したい場合は「@<em>{1truecm}」とする必要があります。しかしここではそこまで厳密な1cmを必要とはしていないので、@<em>{sty/mytextsize.sty}では「@<em>{1cm}」と指定しています。]


=== プログラムコード表示用のフォントを変更

@<LaTeX>{}のデフォルトでは、装飾が多めのフォントがプログラムコードの表示に使われています（@<img>{font_beramono}の上半分）。
このフォントは「@<code>{0}」と「@<code>{O}」や「@<code>{1}」と「@<code>{l}」の区別がつきにくく、また太字にしてもあまり目立たないという欠点があります。

//image[font_beramono][プログラムコードの表示に使われるフォント][scale=0.7]

Starterでは、プログラムコードの表示に使うフォントを、装飾が少ないもの（Bera Mono@<fn>{62vxj}）に変更しています（@<img>{font_beramono}の下半分）。
このフォントは「@<code>{0}」と「@<code>{O}」や「@<code>{1}」と「@<code>{l}」の区別がつきやすく、また太字にしたときも目立つようになっています。
ただし「@<code>$'$」（シングルクォート）と「@<code>$`$」（バッククォート）の区別がつきにくくなっているので注意してください。
//footnote[62vxj][@<href>{http://www.tug.dk/FontCatalogue/beramono/} でサンプルが見れます。]

プログラムコードの表示に向くフォントとしては、他にも「Inconsolata」@<fn>{iii4v}や「Nimbus Mono Narrow」@<fn>{vo5f3}があります。
興味がある人は調べてみてください。
//footnote[iii4v][@<href>{http://www.tug.dk/FontCatalogue/inconsolata/} でサンプルが見れます。]
//footnote[vo5f3][@<href>{http://www.tug.dk/FontCatalogue/nimbus15mononarrow/} でサンプルが見れます。]


=== プログラムコードに枠線をつけることが可能

Starterでは、プログラムコードに枠線をつけるための設定が用意されています。
プログラムコードがページまたぎしている場合は、枠線があったほうが読者にとって分かりやすいです（@<img>{program_border}）。

//image[program_border][プログラムコードがページまたぎした場合][scale=1.0,border=off]

 * 枠線がないと、プログラムコードが次のぺージに続いているかどうかを現在のページだけでは判断できず、次のページを見ないと判断できません。
 * 枠線があると、プログラムコードが次のぺージに続いているかどうかを現在のページだけで判断できます。

JavaScriptの「@<code>$}$」やRubyの「@<code>{end}」があると、プログラムがまだ続いているかどうかの手がかりになります。
しかしPythonのようにインデントでブロックを表すようなプログラミング言語ではそれらのような手がかりがないので、枠線をつけたほうがいいでしょう。

Starterでプログラムコードに枠線をつけるには、@<em>{config-starter.yml}で「@<code>{program_border: true}」を設定してください。


=== 章や節のデザインを変更可能

Starterでは、章(Chapter)や節(Section)のデザインを変更できます。
例を2つ挙げておきます（@<img>{chaphead_design_3}、@<img>{chaphead_design_2}）。

//image[chaphead_design_3][章タイトルをセンタリング、上下に太線、節タイトルに下線][scale=0.7]

//image[chaphead_design_2][章タイトルを右寄せ、下に細線、節タイトルの行頭にクローバー][scale=0.7]

これらのデザインを調整するときは、@<em>{config-starter.yml}で設定を変更してください。
この設定で飽き足らない場合は@<em>{sty/starter.sty}を編集してください。

なおStarterでは、@<img>{chaphead_design_3}のように章タイトルの上下に太い線を入れた場合でも、まえがきや目次やあとがきのタイトルには太い線を入れないようにしています。
これは意図的な仕様です。

また節タイトルが長すぎて2行になることがあるなら、@<em>{config-starter.yml}での「@<code>{section_decoration:}」の値を「@<code>{leftline}」または「@<code>{numbox}」のどちらかにすることを強くお勧めします（@<img>{sechead_design_4}）。
ほかの設定では節タイトルが2行になるとデザインが大きく崩れてしまうので注意してください。

//image[sechead_design_4][節タイトルのデザイン：上が「@<code>{leftline}」、下が「@<code>{numbox}」][scale=0.9,border=on]


=== 章のタイトルページを作成可能

Starterでは、章(Chapter)のタイトルと概要を独立したページにできます（@<img>{chaptitlepage_sample}）。
これは商用の書籍ではよく見かける方法です。

//image[chaptitlepage_sample][章のタイトルと概要を独立したページにした例（章ごとの目次つき）][scale=0.5]

やり方は簡単で、章タイトルと概要を書いたあとに「@<code>$//makechaptitlepage[toc=section]$」と書くだけです。
これで章タイトルページが作られ、背景色がつき、その章の目次もつきます@<fn>{8poyo}。
//footnote[8poyo][実装の詳細は@<em>{sty/starter.sty}の@<em>{makechaptitlepage}コマンドを参照してください。]

//emlist[サンプル]{
= Re:VIEW Starter FAQ

@<letitgo>$//$abstract{
StarterはRe:VIEWを拡張していますが、Re:VIEWの設計にどうしても影響を受けるため、できないことも多々あります。

このFAQでは、「何ができないか？」を中心に解説します。
@<letitgo>$//$}

@<b>$//makechaptitlepage[toc=section]$
//}

注意点が2つあります。

 * 「@<code>$//abstract{ ... }$」は必須です。
   これがないと、「@<code>$//makechaptitlepage$」があっても章タイトルページが作られません。
 * 「@<code>$//makechaptitlepage[toc=section]$」はすべての章に書く必要があります。
   これを書き忘れた章があると、そこだけ章タイトルページが作られません。


=== スペースがあるのに節や項が改ページされるのを避ける

LaTeXでは、ページ最後の段落が最初の1行だけになるを避けようとします。
そのため、もし節や項のタイトル直後が1行だけになりそうになったら、節や項ごと改ページしてしまいます。
そのせいで、余計なスペースが空いてしまうことがあります（@<img>{page_clubline}左）。

Starterでは、ページ最後の段落が最初の1行だけになるのを許します。
そのおかげで、余計なスペースが空くのを避けられます（@<img>{page_clubline}右）。
またこの機能は、@<em>{config-starter.yml}の「@<code>{page_clubline: true}」で変更できます。

//image[page_clubline][スペースがあるのに節や項が改ページされてしまう][scale=1.0]


=== 目次の文字を小さく、行間を狭く

Starterでは、目次のデザインを少し変更しています。

 * 章(Section)の文字をゴシック体にしました。
   項(Subsection)の文字は明朝体のままなので、これで目次での章と項が見分けやすくなります。
 * 項(Subsection)の文字を少し小さくし、行間を狭くしました。
   これにより、目次にとられるページ数を少しだけ減らせます。

目次のデザインを修正する場合は、@<em>{sty/starter.sty}の中で「@<em>{\l@section}」や「@<em>{\l@subsection}」を探して、適宜修正してください。
特に目次のページ数が多い場合は、行間を狭めて（「@<em>{\baselineskip}」を小さくして）みてください。


=== キャプションのデザインを変更

Starterでは、ソースコードや表や図のキャプション（説明）を次のように変更しています。

 * フォントをゴシック体にする
 * 先頭に「▲」や「▼」をつける

これはTechBooster製テンプレートのデザインを参考にしました。
ただし@<LaTeX>{}マクロの定義はまったく別です@<fn>{82e4v}。

//footnote[82e4v][なおこれに関して、「@<code>$\reviewimagecaption$」というコマンドを新たに定義し、「@<code>$\reviewimage$」環境が「@<code>$\caption$」のかわりにそれを使うよう、LATEXBuilder#image_image()にモンキーパッチを適用しています。モンキーパッチはlib/hooks/monkeypatch.rbにあり、review-ext.rbが読み込んでいます。]


=== 引用のデザインを変更

引用「@<code>$//quote{$ ... @<code>$//}$」のデザインを変更し、左側に縦棒がつくようにしました。

Re:VIEWでは@<LaTeX>{}のデフォルトデザインのまま（全体が少しインデントされるだけ）なので、引用であることが分かりにくいです。
これに対し、Starterでは左側に縦棒がつくので、引用であることがより分かりやすくなっています。

また引用中に複数の段落を入れた場合、段落の先頭が1文字分インデントされます（Re:VIEW標準ではインデントされません）。

//emlist[サンプル]{
@<letitgo>$//$quote{
その者蒼き衣を纏いて金色の野に降りたつべし。
失われし大地との絆を結び、ついに人々を清浄の地に導かん。
@<letitgo>$//$}
//}

//sampleoutputbegin[表示結果]

//quote{
その者蒼き衣を纏いて金色の野に降りたつべし。
失われし大地との絆を結び、ついに人々を清浄の地に導かん。
//}

//sampleoutputend




=== ページヘッダーを変更

一般の書籍では、ページヘッダーは次のような形式になっています。

 * 見開きで左のページのヘッダーには、章タイトルを表示
 * 見開きで右のページのヘッダーには、節タイトルを表示

しかしRe:VIEWでは、両方のページのヘッダーに章タイトルと節タイトルが表示されています。
これはおそらく、タブレットのような見開きがない閲覧環境を想定しているのだと思います。

Starterではこれを変更し、一般の書籍と同じようなヘッダーにしています。
ただしタブレット向けの場合は、Re:VIEWと同じようにしています。


=== ページ番号のデザインを変更

Re:VIEWのデフォルトでは、ページ番号はたとえば「10」のように表示されるだけです。

Starterでは、ページ番号を「@<embed>$--$ 10 @<embed>$--$」のような表示に変更しています。
これは、ページ番号であることをより分かりやすくするためです。
詳しくは@<em>{sty/starter.sty}を参照してください。


=== 箇条書きの行頭記号を変更

@<LaTeX>{}では、箇条書きの行頭に使われる記号が、第1レベルでは小さい黒丸「@<embed>{|latex|$\bullet$}」、第2レベルではハイフン「@<embed>{|latex|--}」でした。

//embed[latex]{
{\renewcommand{\labelitemii}{--}
//}

//emlist[サンプル]{
 * 第1レベル
 ** 第2レベル
//}

//sampleoutputbegin[表示結果（変更前）：]

 * 第1レベル
 ** 第2レベル

//sampleoutputend



//embed[latex]{
}
//}

しかしこれだと、箇条書きの記号ではなくマイナス記号に見えてしまいます。

Starterではこの第2レベルの記号を、小さい白丸「@<embed>{|latex|$\circ$}」に変更しました。
これで、より自然な箇条書きになりました。

//sampleoutputbegin[表示結果（変更後）：]

 * 第1レベル
 ** 第2レベル

//sampleoutputend




==={tyly6} 文章中のコードに背景色をつけられる

Starterでは、「@<code>$@<nop>{@}<code>{...}$」を使って文章中に埋め込んだソースコードに背景色（薄いグレー）をつけられます。
そのためには、@<em>{config-starter.yml}で「@<code>$inlinecode_gray: true$」を設定してください。

//list[][ファイル「@<em>{config-starter.yml}」]{
  inlinecode_gray: true
//}

こうすると、@<em>{sty/starter.sty}において以下の@<LaTeX>{}マクロが有効になります。
背景色を変えたい場合はこのマクロを変更してください。

//list[][ファイル「@<em>{sty/starter.sty}」]{
  \renewcommand{\reviewcode}[1]{%
    \,%                        % ほんの少しスペースを入れる
    \colorbox{shadecolor}{%    % 背景色を薄いグレーに変更
      \texttt{#1}%             % 文字種を等幅フォントに変更
    }%
    \,%                        % ほんの少しスペースを入れる
  }
//}


=== 表紙用PDFファイル

Starterでは、表紙用のPDFファイルを本文のPDFに挿入できます。
@<em>{config.yml}に以下のような設定をしてください。

//list[][config.yml（下のほう）]{
pdfmaker:
  ....(省略)....
  coverpdf_files: [cover.pdf]  # PDFファイル名（複数指定可）
  coverpdf_option: "offset=-2.3mm 2.3mm"  # 必要に応じて微調整
//}

いくつか注意点があります。

 * 電子書籍用PDFでのみ挿入されます。印刷用PDFでは挿入されません@<fn>{w0spr}。
   電子書籍用PDFを生成する方法は@<secref>{bn2iw}を参照してください。
 * 表紙画像はPDFのみ対応です。PNGやJPGは対応していません。
 * 挿入位置がずれる場合は、「@<code>{coverpdf_option:}」の設定を調整してください。
 * 複数のPDFファイル名は「@<code>{[cover1.pdf, cover2.pdf]}」のように指定します（「@<code>{,}」のあとに半角空白が必要なことに注意）。

//footnote[w0spr][印刷所に入稿する場合、通常は表紙は本文とは別のPDFファイルにします。そのため、印刷用PDFには表紙をつけません。]

//note[PNGやJPGの画像をPDFに変換する]{

macOSにてPNGやJPGをPDFにするには、画像をプレビュー.appで開き、メニューから「ファイル > 書き出す... > フォーマット:PDF」を選んでください。

macOS以外の場合は、「画像をPDFに変換」などでGoogle検索すると変換サービスが見つかります。

//}

//note[表紙画像を他のソフトウェアで挿入する]{

表紙画像を挿入するのは、Re:VIEW Starterや@<LaTeX>{}ではなく他のソフトウェアですると手軽です。
macOSなら「プレビュー.app」を使えば、PDFに表紙画像を入れたりタイトルページを差し替えるのが簡単にできます。

1つの道具で何でも行うのではなく、目的に応じて道具を変えることを検討してみてください。

//}


=== タイトルページと奥付を独立したファイルに

Starterでは、タイトルページ（「大扉」といいます）と、本の最終ページにある「奥付」を、それぞれ別ファイルに分離しました。

 * @<em>{sty/mytitlepage.sty} … タイトルページを表します。
 * @<em>{sty/mycolophon.sty} … 奥付を表します。

タイトルページや奥付のデザインが気に入らない場合は、これらを編集してください。


=== 奥付が必ず最終ページになるよう修正

Re:VIEWでは、奥付のページは単に改ページされてから作成されます。
そのため、場合によっては奥付がいちばん最後のページではなく、最後から2番目のページになることがあります（この場合、最後のページは空白ページになります）。

Starterではこれを改善し、奥付が必ず最後のページになるようにしています。
詳しくは@<em>{sty/starter.sty}の「@<code>$\reviewcolophon$」コマンドを参照してください。
この@<LaTeX>{}コマンドは@<em>{sty/mycolophon.sty}から呼び出されています。


=== コラムがページまたぎする場合は横線を入れない

Starterでは、コラムが長くてページをまたいでしまう場合に、横線をいれないようにしています（@<img>{column_openframe}）。
こうすると、コラムが続いていることが分かりやすいです。

//image[column_openframe][コラムがページをまたぐときに横線を入れない][scale=0.8]


=== リンクテキストのURLを脚注に記載

Re:VIEWでたとえば@<br>{}
@<code>$@<nop>{@}<href>{https://pripri-anime.jp/, プリンセス・プリンシパル}$@<br>{}
のように書くと、ちょうどHTMLにおける@<br>{}
@<code>$<a href="https://pripri-anime.jp/">プリンセス・プリンシパル</a>$@<br>{}
のようなリンクテキストになります。
しかしHTMLならリンク先のURLを調べられますが、PDFにして印刷するとリンクのURLが何だったのか、読者には分かりません。

そこでStarterでは、PDFの場合はリンクテキストのURLを脚注に記載するようにしました。
たとえば「@<code>$@<nop>{@}<href>{https://pripri-anime.jp/, プリンセス・プリンシパル}$」は「@<href>{https://pripri-anime.jp/, プリンセス・プリンシパル}」のように表示されます。

また脚注の中のリンクテキストでは、脚注を使わないようにしています。
たとえば「@<code>$//footnote[pripri][@<nop>{@}<href>{https://pripri-anime.jp/, プリンセス・プリンシパル}の劇場アニメは2020年4月公開！$」は脚注の中にリンクテキストがあるので、このページ下の脚注のように表示されます@<fn>{pripri}。

//footnote[pripri][@<href>{https://pripri-anime.jp/, プリンセス・プリンシパル}の劇場アニメは2020年4月公開！]

なおこの挙動は、@<em>{config-starter.yml}の「@<code>{linkurl_footnote: @<i>{bool}}」で制御できます。
この値が@<code>{true}ならURLを脚注に記載し、@<code>{false}ならしません。


=={sec-sty} @<LaTeX>{}のコマンドやスタイルファイルに関する機能


=== Dockerコマンドを簡単に起動するタスクを追加

Docker環境を使ってPDFをコンパイルするには、次のようなコマンドの入力が必要です。

//terminal[][Docker環境を使ってPDFをコンパイルする]{
$ docker run --rm -v $PWD:/work kauplan/review2.5 /bin/bash -c "cd /work; rake pdf"
//}

しかしこれは長いので、「@<code>{rake docker:pdf}」でコンパイルできるようにしました。

//terminal[][より簡単にDocker環境でPDFをコンパイルする]{
$ rake docker:pdf
//}

他にも次のようなrakeタスクを用意しました。

 * @<code>{rake docker:pull} ： @<code>{docker pull kauplan/review2.5}を実行する。
 * @<code>{rake docker:pdf} ： Docker経由でPDFを生成する。
 * @<code>{rake docker:pdf:nombre} ： Docker経由でPDFにノンブルをつける。
 * @<code>{rake docker:epub} ： Docker経由でePubページを作成する。
 * @<code>{rake docker:web} ： Docker経由でWebページを作成する。

なおこれらのタスクでは、「@<tt>{STARTER_}」で始まる環境変数も継承されます。


=== スタイルシートを追加

Starterでは、次のような独自のスタイルファイルを追加しています。

 : sty/starter.sty
	Starterのサイトで選択したオプションに従って生成されたスタイルファイルです（Starterのバージョンが上がるたび、このファイルもよく変更されます）。
	デザインを調整したい場合などはこのファイルを編集するか、後述の@<em>{sty/mystyle.sty}で上書きしてください。
 : sty/mytextsize.sty
	本文の幅やページ左右の余白を設定するためのスタイルファイルです。
	PDFのサイズ（B5やA5）を変更する場合は、@<em>{config.yml}の「@<code>$texdocumentclass:$」を変更してください。
 : sty/mystyle.sty
	ユーザ独自の@<LaTeX>{}マクロ（コマンドや環境）を追加したり、既存のマクロを上書きするためのファイルです。
	中身は空なので、自由に追加や上書きしてください。
 : sty/mytitlepage.sty
	タイトルページ（大扉）の内容を表すスタイルファイルです。
	デザインが気に入らない場合は編集してください。
 : sty/mycolophon.sty
	最終ページの「奥付」を表すスタイルファイルです。
	デザインが気に入らない場合は編集してください。

@<em>{sty/mytextsize.sty}と@<em>{sty/starter.sty}は、どちらも自動生成されます。
なので同じファイルにできそうですが、読み込むタイミングが異なるため、別ファイルにしています。

 * @<em>{sty/mytextsize.sty}は本文の幅や高さを指定するので、他のスタイルファイルより先に読み込まれます。
 * @<em>{sty/starter.sty}は既存の@<LaTeX>{}マクロ（コマンドや環境）を上書きするので、他のスタイルファイルより後に読み込まれます。


==={bn2iw} 印刷用PDFと電子用PDFを切り替える

Starterには、印刷用PDFと電子用PDFを切り替えて出力する機能があります@<fn>{ctr1x}。
//footnote[ctr1x][ただしタブレット用にプロジェクトを作成した場合は、切り替えは無意味です。]

 : 印刷用PDF
	紙に印刷するためのPDFです。色はモノクロで、またA5の場合はページ左右の余白幅を変更します。
 : 電子用PDF
	ダウンロードで配布するためのPDFです。色はカラーで、ページ左右の余白は同じ幅です。

//note[ページ左右の余白幅を変える理由]{

印刷用PDFにおいてページ左右の余白幅を変更するのは、本の読みやすさを保ったまま1行あたりの文字数を増やすためです。B5の場合はたいてい十分な紙幅があるので、ページ左右の余白幅は同じままで大丈夫です。A5の場合は見開きで内側の余白幅を確保したまま、外側の余白幅を狭めることで、1行あたりの文字数を増やします。
詳しくは@<secref>{jwevu}を参照してください。

当然ですが、このような変更は電子用PDFでは必要ありません。
//}

設定ファイル@<code>{config-starter.yml}の中にある「@<code>{starter: target: }@<i>{xxx}」の値が「@<code>{pbook}@<fn>{0mxlb}」だと印刷用PDFが、「@<code>{ebook}@<fn>{4545b}」だと電子用PDFが生成されます。初期設定では「@<code>{pbook}」になっているので、デフォルトでは印刷用PDFが生成されます。
//footnote[0mxlb][「pbook」はprinting bookの略です。]
//footnote[4545b][「ebook」はelectric bookの略です。]

またこの値は環境変数@<fn>{4r53n}@<code>{$STARTER_TARGET}で上書きできます。具体的には次のようにすると印刷用と電子用のPDFを切り替えられます。
//footnote[4r53n][環境変数とは、コマンドプロセスが参照する外部変数です。環境変数を設定することでコマンドの挙動を一部変更できます。詳しくは「環境変数」でぐぐってください。]

//terminal[][印刷用PDFと電子用PDFを切り替える]{
### 印刷用PDFを生成（デフォルト）
$ rake pdf    # または STARTER_TARGET=pbook rake pdf

### 電子用PDFを生成（環境変数を使って設定を上書き）
$ STARTER_TARGET=ebook rake pdf
//}

ただしこの機能では、@<LaTeX>{}のスタイルファイル（@<code>{sty/starter.sty}や@<code>{sty/mytextsize.sty}）の中で行える範囲でしか変更はできません。それ以上のことがしたい場合は、@<secref>{chap02-faq|qvtlq}を参照してください。


==={vb2z3} @<LaTeX>{}コマンドの実行回数を減らす

Re:VIEWでは、@<LaTeX>{}コマンド（@<code>{uplatex}）の実行を3回行います。

 * 1回目でページ番号が決まる。
 * 2回目で目次などを生成する（これでページ番号がずれることがある）。
 * 3回目で索引をつける。

Starterではこれを変更し、1回または2回の実行で済むようにしました@<fn>{25e43}。
このおかげで、ページ数が多いときのコンパイル時間が大きく短縮されます。
ただし索引を作成する場合はコンパイル回数が1回増えます。

//footnote[25e43][これはauxファイルとtocファイルを保存することで実現しています。またそのために@<code>{ReVIEW::PDFMaker}クラスを全面的に書き換えました。]


==={oivuz} 指定した章だけをコンパイルする

Starterでは、環境変数「@<code>{$STARTER_CHAPTER}」を設定するとその章(Chapter)だけをコンパイルします。
これは章の数が多い場合や、著者が多数いる合同誌の場合にはとても有効です。

//terminal[][例：chap02-faq.reだけをコンパイルする]{
bash$ export STARTER_CHAPTER=chap02-faq   # 「.re」はつけない
bash$ rake pdf  # またはDockerを使っているなら rake docker:pdf
//}

このとき、他の章は無視されます。
また表紙や目次や大扉や奥付も無視されます。

全体をコンパイルする場合は、「@<code>{$STARTER_CHAPTER}」をリセットしてください。

//terminal[][全体をコンパイルする]{
bash$ unset STARTER_CHAPTER    # 「$」はつけないことに注意
//}


==={8v2z5} ドラフトモードにして画像読み込みを省略する

Starterでは画像の読み込みを省略する「ドラフトモード」を用意しました。
ドラフトモードにすると、画像のかわりに枠線が表示されます。
こうすると、（@<LaTeX>{}のコンパイル時間は変わりませんが）DVIファイルからPDFを生成する時間が短縮されます。

この機能は、図やスクリーンショットが多い場合や、印刷用に高解像度の画像を使っている場合は、特に効果が高いです。

ドラフトモードにするには、@<em>{config-starter.yml}で「@<code>{draft: true}」を設定するか、または環境変数「@<em>{$STARTER_DRAFT}」に何らかの値を入れてください。

//terminal[][ドラフトモードにしてPDFを生成する]{
$ export STARTER_DRAFT=1  # ドラフトモードをonにする
$ rake pdf                # またはDocker環境なら rake docker:pdf

$ unset STARTER_DRAFT     # ドラフトモードをoffにする
//}

また「ドラフトモードにしてPDF生成時間を短縮したい、でもこの画像は表示して確認したい」という場合は、「@<code>$//image[][][draft=off]$」のように第3引数に@<code>{draft=off}を指定すると、その画像はドラフトモードが解除されてPDFに表示されます。


=== コンパイル時の出力を抑制

@<LaTeX>{}でコンパイルすると（つまり@<em>{uplatex}コマンドを実行すると）、通常ではたくさんのメッセージが出力されます。
これはとても煩わしいので、Starterでは出力を抑制するために@<em>{uplatex}コマンドに「@<code>{-interaction=batchmode}」オプションをつけています。

しかしこのオプションをつけると、今度はエラーメッセージが表示されないという問題があります。
つまり、こういうことです：

 * 出力を抑制したいなら、@<LaTeX>{}コマンドに「@<code>{-interaction=batchmode}」オプションをつける。
 * しかし「@<code>{-interaction=batchmode}」オプションをつけると、エラーメッセージが表示されない。

なんというクソ仕様でしょう！
このクソ仕様を、Starterでは次のように回避しています。

 1. 「@<code>{-interaction=batchmode}」オプションをつけてコンパイルする。
 2. エラーになったら（つまりコマンドの終了ステータスが0でなければ）、「@<code>{-interaction=batchmode}」オプションを@<strong>{つけずに}コンパイルし直すことで、エラーメッセージを表示する。

今のところ、この方法がいちばん妥当でしょう。

なおこの変更は「@<em>{rake pdf}」または「@<em>{rake docker:pdf}」コマンドでのみ行われます@<fn>{g6wy6}。
「@<em>{review-pdfmaker config.yml}」を実行した場合はもとの挙動（つまりコンパイルメッセージがたくさん出る）になるので注意してください。
//footnote[g6wy6][実装の詳細は@<em>{lib/tasks/review.rake}を参照してください。]

ちなみに、@<LaTeX>{}のコマンドはエラーメッセージを標準エラー出力(stderr)に出してくれません。
クソかよ。


=== @<LaTeX>{}コマンドにオプションを追加

Starterでは、@<LaTeX>{}コマンド（@<em>{uplatex}）に以下のオプションをつけています。

: @<code>{-halt-on-error}
	@<LaTeX>{}のコンパイルエラー時に、インタラクティブモードにせず、そのままコマンドを終了させるオプションです。
: @<code>{-file-line-error}
	@<LaTeX>{}のコンパイルエラー時に、エラー発生箇所の行番号に加えて、ファイル名も出力するようにするオプションです。

指定箇所は@<em>{config.yml}の「@<code>$texoptions:$」です。


=== 実行する@<LaTeX>{}コマンドをオプションつきで出力

Starterでは、実行する@<LaTeX>{}コマンドをオプションつきで出力するように変更しています@<fn>{7a93y}。
こうすることで、特にエラーが発生した場合にどんなコマンドを実行したかを調べるのに役立ちます。
//footnote[7a93y][この変更は、@<em>{lib/tasks/review.rake}で定義されている「@<code>{pdf}」タスクを書き換えることで実現しています。]

ただしこれは「@<code>{rake pdf}」または「@<code>{rake docker:pdf}」を実行したときだけであり、コマンドラインから直接「@<code>{review-pdfmaker config.yml}」を実行したときは出力されません@<fn>{orc59}ので注意してください。
//footnote[orc59][なぜなら、この変更は「@<code>{pdf}」タスクを書き換えることで実現しているので、@<em>{review-pdfmaker}コマンドには影響しないからです。]

次が実行例です。
@<em>{uplatex}コマンドや@<em>{dvipdfmx}コマンドが、オプションつきで出力されていることが分かります。

//terminal[][実行例]{
$ rake pdf
compiling chap00-preface.tex
compiling chap01-starter.tex
compiling chap02-review.tex
compiling chap99-postscript.tex

[review-pdfmaker]$ /usr/bin/ruby /tmp/xxx-book/lib/hooks/beforetexcompile.rb /tmp/xxx-book/xxx-book-pdf /tmp/xxx-book

[review-pdfmaker]$ uplatex -halt-on-error -file-line-error -interaction=batchmode samplebook.tex
This is e-upTeX, Version 3.14159265-p3.8.1-u1.23-180226-2.6 (utf8.uptex) (TeX Live 2018) (preloaded format=uplatex)
 restricted \write18 enabled.
entering extended mode

[review-pdfmaker]$ uplatex -halt-on-error -file-line-error -interaction=batchmode samplebook.tex
This is e-upTeX, Version 3.14159265-p3.8.1-u1.23-180226-2.6 (utf8.uptex) (TeX Live 2018) (preloaded format=uplatex)
 restricted \write18 enabled.
entering extended mode

[review-pdfmaker]$ uplatex -halt-on-error -file-line-error -interaction=batchmode samplebook.tex
This is e-upTeX, Version 3.14159265-p3.8.1-u1.23-180226-2.6 (utf8.uptex) (TeX Live 2018) (preloaded format=uplatex)
 restricted \write18 enabled.
entering extended mode

[review-pdfmaker]$ dvipdfmx -d 5 -z 3 book.dvi
book.dvi -> book.pdf
[1][2][3][4][5][6][7][8][9][10][11][12]
386603 bytes written
//}

なお実行結果を見ると、@<LaTeX>{}のコンパイル（つまりuplatexコマンドの実行）が3回行われていることが分かります。
これはバグではなく、Re:VIEWの仕様です。
理由は、ページ数に変更があっても対応できるようにするためと思われます。


=== PDF変換を高速化する

DVIファイルをPDFファイルに変換する「@<em>{dvipdfmx}」コマンドのオプションを、圧縮率を少し下げるかわりに短時間で終わるようにするよう、設定しました。

具体的には、@<em>{config.yml}の「@<code>$dvioptions:$」という項目を、Re:VIEWのデフォルトの「@<code>$"-d 5 -z 9"$」から「@<code>$"-d 5 -z 3"$」に変更しています。
「@<code>$-z 9$」は圧縮率を最大にするので時間がかかり、「@<code>$-z 3$」は圧縮率を下げるかわりに短時間で済みます。

PDFファイルのサイズを少しでも減らしたい場合は、「@<code>$-z 9$」にしてみてください。


=== PDFにノンブルをつける

印刷所によっては、PDFにノンブルをつけるのが必須です。
たとえば日光企画@<fn>{lmknh}さんは、ノンブルをつけないと入稿ができません@<fn>{26k16}。
//footnote[lmknh][技術書典でいちばん多くのサークルがお世話になっている印刷所。電話対応のお姉さんがとても親切。入稿ページの使い方が分かりにくいので、ほとんどの初心者はお姉さんのお世話になるはず。]
//footnote[26k16][@<href>{http://www.nikko-pc.com/q&a/yokuaru-shitsumon.html#3-1}より引用：『ノンブルは全ページに必要です。ノンブルが無いものは製本時にページ順に並び替えることが非常に困難な為、落丁・乱丁の原因となります。』]

//note[■ノンブルとは]{

ノンブルとは、すべてのページにつけられた通し番号です。
ページ番号と似ていますが、ページ番号が読者のための番号なのに対し、ノンブルは印刷所の人が間違えずに作業するための番号です。
具体的には次のような違いがあります。

 * ページ番号は読者のためにつけるので、読者から見えやすい場所につける。
   ノンブルは印刷所の人が見えればいいので、読者には見えにくい場所につける。
 * ページ番号は、まえがきや目次では「i, ii, iii, ...」、本文では「1, 2, 3, ...」と増える。
   ノンブルは最初から「1, 2, 3, ...」と増える。
 * ページ番号は、タイトルページや空白ページではつかないことがある。
   ノンブルは、すべてのページに必ずつける必要がある。

詳しくは「ノンブル」でGoogle検索してください。

//}

Starterでは、PDFにノンブルをつけるためのrakeタスク「@<em>{pdf:nombre}」@<fn>{nqjz9}を用意しています。
//footnote[nqjz9][@<em>{lib/tasks/starter.rake}で定義されています。]

//terminal{
$ gem install combine_pdf  # 事前作業（最初の1回だけ）
$ rake pdf:nombre          # Docker環境なら rake docker:pdf:nombre
//}

これで、PDFファイルにノンブルがつきます。

もし@<em>{pdf:nombre}タスクがうまく動作しない場合は、かわりに@<href>{https://kauplan.org/pdfoperation/}を使ってください。


=== rakeコマンドのデフォルトタスクを指定する

Re:VIEWでは、rakeのデフォルトタスクが「@<em>{epub}」になっています。
つまり引数なしでrakeコマンドを実行すると、epubを生成するタスクが実行されます。

これはあまり便利とはいえないし、なによりRubyとrakeをよく知らない人にとっては優しくない仕様です。

そこでStarterでは、rakeの使い方を表示する「@<em>{help}」タスクを用意し、これをデフォルトタスクにしています。
このおかげで、引数なしでrakeコマンドを実行するとrakeの使い方が表示されます。
このほうが、Rubyとrakeをよく知らない人にとって優しいでしょう。

//terminal[][引数なしでrakeコマンドを実行すると、rakeの使い方が表示される][fold=off]{
$ rake
rake -T
rake all                # generate PDF and EPUB file
rake clean              # Remove any temporary products
rake clobber            # Remove any generated files
rake docker:epub        # + run 'rake epub' on docker
rake docker:pdf         # + run 'rake pdf' on docker
rake docker:pdf:nombre  # + run 'rake pdf:nombre' on docker
rake docker:setup       # + pull docker image for building PDF file
rake docker:web         # + run 'rake web' on docker
rake epub               # generate EPUB file
rake help               # + list tasks
rake html               # build html (Usage: rake build re=target.re)
rake html_all           # build all html
rake images             # + convert images (high resolution -> low resolution)
rake images:toggle      # + toggle image directories ('images_{lowres,highres}')
rake pdf                # generate PDF file
rake pdf:nombre         # + add nombre (rake pdf:nombre [file=*.pdf] [out=*.pdf])
rake preproc            # preproc all
rake web                # generate stagic HTML file for web
//}

上の表示結果のうち、コマンドの説明文の先頭に「@<code>{+}」がついているのが、Starterで独自に用意したタスクです。

また環境変数「@<em>{$RAKE_DEFAULT}」を設定すると、それがデフォルトタスクになります。
たとえば「@<em>{pdf}」タスクをデフォルトにしたい場合は、次のようにします。

//terminal[][pdfタスクをデフォルトタスクにする]{
$ export RAKE_DEFAULT=pdf    # デフォルトタスクを変更する。
$ rake                       # 引数がないのにpdfタスクが実行される。
compiling chap00-preface.tex
compiling chap01-starter.tex
compiling chap02-review.tex
compiling chap99-postscript.tex

[review-pdfmaker]$ uplatex -halt-on-error -file-line-error -interaction=batchmode samplebook.tex
....(以下省略)....
//}
