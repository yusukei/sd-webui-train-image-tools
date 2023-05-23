# sd-webui-train-image-tools
学習用の画像作成のための補助ツールです。

![](docs/ui.png)

## 機能
### Single
１枚の画像を処理します。

### Batch from Dir
入力元と出力先のディレクトリを指定して、バッチ処理を行います。

### Remove Background

- Enable\
画像から背景を削除して人物のみにします。

- Background Color\
切り抜いた背景の部分の色を設定します。\
透明、白、黒から選択します。

- Model\
背景を削除する際のモデルを指定します。

- CPU Only\
背景を削除する際にCPUのみで処理をするか指定します。\
GPUを使用するにはonnxruntime-gpuが正しく動作するように環境を設定しておく必要があります。


### Crop a face
- Enable\
画像から顔を検出して切り出します。

- Face Type\
検出の対象を写真向けにするか、アニメ向けにするかを選択します。

- Face Padding\
検出した顔を切り出す際の余白部分のサイズです。

### Crop to Square
- Enable\
画像を正方形になるようにクロップします。
