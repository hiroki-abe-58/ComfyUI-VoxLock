# VoxLock Voice Isolation for ComfyUI（日本語概要）

公式の **VoxLock** モデル（[arbazsiddiqui/voxlock](https://github.com/arbazsiddiqui/voxlock)）で、録音から一人の声だけを残し、
重なって話している他の人の声を取り除く ComfyUI ノードパックです。残したい声の数秒（ボイスサンプル）を与えるか、自動で見つけさせます。
ONNX Runtime で CPU だけで動き、この環境では実時間の約 12 倍の速さでした。ノードは公式の `voxlock` パッケージを呼び出し、何も
ダウンロードしません。

> **非公式**のインテグレーションです。VoxLock の作者、Comfy Org とは関係がなく、承認も受けていません。
> 詳細は英語版 [README.md](README.md) と [docs/VERIFICATION.md](docs/VERIFICATION.md) を参照してください。

## ノード

| ノード | 内容 |
| --- | --- |
| **VoxLock Loader** | モデルの精度（int8 / int8-full / fp16 / fp32）と CPU スレッド数を選ぶ |
| **VoxLock Isolate Voice** | `AUDIO` → 残した声だけの `AUDIO` と JSON レポート。任意入力 `voice_sample` に残したい声の数秒を渡せる |

前に ComfyUI 標準の **Load Audio**、後ろに **Save Audio** / **Preview Audio** をつなぎます。同梱ワークフローもこの構成です。

- ボイスサンプルがない場合は、公式コマンドラインと同じく入力の最初の 15 秒のうち最も大きい 3 秒の声を残します。冒頭で別の人の声の
  ほうが大きいときはサンプルを渡してください。
- モデルは 16 kHz モノラルの音声を扱います。どのサンプリングレート・チャンネル数の入力も、公式の `voxlock.load_audio` と同じ方法で
  モノラル化・リサンプルします。出力は 16 kHz（`output_sample_rate` = `16000`）か、入力のレートに戻したもの（`input`。内容は
  16 kHz 帯域のまま）です。
- 出力は入力と同位相で、レベルもほぼ同じなので、他のトラックと重ねて使えます。

## 確認した範囲

Windows 11、ComfyUI v0.38.0、CPU（Intel Core Ultra 9 285K）：

- **公式パッケージとビット単位で一致**：実際の音声（VoxLock リポジトリのデモ混合音 3 本、4 種類すべての精度、48 kHz ステレオ版、
  ボイスサンプル指定）で、同じファイルに対する公式 `voxlock` 1.1.1 の出力と float サンプルの SHA-256 が 8 件中 8 件一致しました。
  公式側は `pip install voxlock==1.1.1` だけの別環境で実行しています。
- **実際の ComfyUI キュー**：Load Audio → VoxLock → Save Audio（FLAC）の出力は、5 件すべてで公式出力と 16 ビットの半ステップ以内
  （16 ビット FLAC の丸め）で一致しました。
- **CI**（GitHub Actions、Ubuntu と Windows、Python 3.12 / torch 2.14.1 と Python 3.10 / torch 2.8.0）：合成した 2 人の声で実モデルを
  動かし、全精度で公式手順と比較、ComfyUI 自身のワークフロー検証、実際の ComfyUI キュー。

**未確認**：CI 以外の Linux / macOS、他の ComfyUI バージョン、音楽や歌声（VoxLock は会話音声向け）、分離品質そのものの評価
（公式出力の再現までを確認しています）。

## インストール

1. V3 ノード API のある ComfyUI（**v0.38.0** で確認）、Python 3.10 以上。GPU は不要です。
2. このリポジトリを `ComfyUI/custom_nodes/` に clone し、ComfyUI の Python に依存パッケージを入れます：

   ```
   git clone https://github.com/hiroki-abe-58/ComfyUI-VoxLock custom_nodes/ComfyUI-VoxLock
   python -m pip install -r custom_nodes/ComfyUI-VoxLock/requirements.txt
   ```

3. モデルをダウンロードします（4 精度すべてで約 275 MB。`--precisions int8` などで選択可）。voxlock 1.1.1 自身が使うリビジョンに固定し、
   全ファイルの SHA-256 を検証します：

   ```
   python custom_nodes/ComfyUI-VoxLock/tools/download_model.py --models-dir models/voxlock
   ```

4. ComfyUI を再起動し、`workflows/gui/isolate_voice.json` または `workflows/gui/isolate_with_sample.json` を開いて、Load Audio で
   自分の録音を選びます。

## 速度とスレッド

16 kHz の音声 10 分、CPU 24 スレッド：int8 は 49.1 秒（実時間の 12 倍）、int8-full は 45.9 秒、fp16 / fp32 は約 73 秒（8 倍）。
`threads` = 0（既定、公式と同じ）は ONNX Runtime に任せます。スレッド数を変えると int8 の出力がわずかに変わります（デモ音声で最大
0.0025、約 -52 dBFS）。同じ設定・同じマシンなら毎回同じ出力です。

## ライセンス

- このパックのコード：**MIT**（[LICENSE](LICENSE)）。
- `voxlock` パッケージ（PyPI からインストール）と VoxLock のモデルファイル（`tools/download_model.py` を実行したときだけ取得）：
  **Apache-2.0**、作者 Arbaz Siddiqui（[NOTICE](NOTICE)）。
