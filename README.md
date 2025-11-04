# Chariloto Pipeline

このリポジトリは、チャリロト (Chariloto) から競輪データを取得し、学習・予測・買い目生成までを行う Python 3.10+ 向けのツール群です。`pip install -r requirements.txt` で依存関係をインストールすると、コマンドラインから以下の処理を一通り実行できます。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # Windows は .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

※ LightGBM を利用したい場合は別途 `pip install lightgbm` を実行してください。インストールできない環境でもランダムフォレストに自動フォールバックします。

## コマンド一覧

すべてのコマンドは UTF-8 (UTF-8-SIG) で入出力され、Windows PowerShell でも文字化けしません。

### 1. 過去データの取得 (`fetch`)

指定期間のレースカードと結果を取得し、学習用 `races.csv` を日付フォルダごとに出力します。

```bash
python main.py fetch \
  --from 2024-01-01 --to 2024-01-31 \
  --out data/fetch_202401 \
  --rate-limit 0.5 --retries 3 --timeout 10
```

出力例 (`data/fetch_202401/20240101/`):

- `cards_info.csv` / `cards_entries.csv`
- `results_info.csv` / `results_entries.csv`
- `payouts.csv`
- `races.csv` (学習用正規化データ)

### 2. モデル学習 (`train`)

複数日の `races.csv` を結合してから学習します。

```bash
python main.py train --races data/races_2024.csv --out model_202410
```

`model_202410/` 配下には `model.joblib` と `metadata.json` が保存されます。

### 3. 今日の出走表と予測 (`today`)

指定（または当日）の日付のレースカードを取得し、学習済みモデルで勝率を推論します。

```bash
python main.py today --model model_202410 --out out --date 2024-04-01
```

出力例 (`out/2024-04-01/`):

- `cards_info.csv`
- `cards.csv`
- `predictions_2024-04-01.csv`

`--midnight-only` を付けると、ミッドナイト開催のみを対象にします。

### 4. 買い目生成 (`bets`)

`bets_from_csv.py` を直接実行するか、`main.py bets` サブコマンドで予算・配分ポリシー付きの買い目を生成できます。

```bash
python bets_from_csv.py \
  --cards out/2024-04-01/cards.csv \
  --pred out/2024-04-01/predictions_2024-04-01.csv \
  --out out/2024-04-01/today_bets.csv \
  --budget 10000 --policy flat --ev-th 1.0
```

`--policy` は `flat`, `proportional`, `kelly` を選択可能です。`--ev-th` で期待値フィルタを調整します。

## モジュール構成

- `main.py`: CLI ハブ。`fetch / train / today / bets` の各サブコマンドを提供。
- `providers/chariloto.py`: 開催スケジュールを巡回し、レース ID を抽出。
- `scrape/chariloto_cards.py`: 出走表のスクレイピングと正規化。
- `scrape/chariloto_results.py`: 結果・払戻情報のスクレイピングと正規化。
- `scrape/normalize.py`: `races.csv` を生成。
- `model.py`: LightGBM / RandomForest による学習・推論。
- `bets_from_csv.py`: `cards.csv` と `predictions.csv` から買い目を生成。
- `utils.py`: ログ・HTTP セッション・CSV 保存などの共通処理。
- `tests/`: HTML モックを用いたユニットテスト。

## テスト

```bash
pytest
```

主要機能（スクレイピング正規化、学習、買い目生成）の最小ケースをカバーしています。

## 注意事項

- 公開サイトの構造変更に備えて、HTML のテキスト・カラム名を複数候補で検出するよう実装しています。
- 実際のスクレイピングではレートリミット・リトライを掛け、例外が発生しても処理を継続します。
- 生成される CSV はすべて `encoding='utf-8-sig'` で保存されます。
