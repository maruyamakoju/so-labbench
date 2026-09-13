# 公開資産トラック（設計ドラフト・未登録）

作成: 2026-09-12 夜 / 状態: **方向はユーザー承認済み（21:20、RQ0はリモコン課題ごと終了）**。課題選定と機材購入が未決
方針（ユーザー、2026-09-12）: **公開モデルを公開データで微調整し、自分のアームは評価装置としてだけ使う。** 自分の60本デモを主語にした実験はRQ0で閉じる。

## 1. 調査で分かった事実

### 環境のギャップ

| | 手元 | 現行 |
|---|---|---|
| lerobot | v0.3.3+10（2025-08-29） | **v0.6.1（2026-08-03）** |
| Python | 3.11 | **3.12以上** |
| データ形式 | v2.1 | **v3.0**（1ファイルに複数エピソード、meta/episodes はparquet） |
| 方策 | act / diffusion / pi0 / pi0fast / smolvla / … | 上記＋ **pi05**、GR00T等は外部 |

公開データの大半が v3.0 なので、**手元の環境では読めない**。逆に現行環境は v2.1 を読まない（変換器は v2.1→v3.0 の一方向のみ）。Windowsでは torchcodec が torch 2.8 以上で使え、なければ pyav に自動で落ちる。

### 公開データセット（SO-101・実機）

| データセット | 本数 | 形式 | カメラ | 内容 | 備考 |
|---|---|---|---|---|---|
| **armnet/armnetbench_v01_lerobot_so101** | 2,499（参照デモ 50本×8タスク＋7方策の評価ロールアウト） | v3.0, 20fps | front 576×1024 / top 576×1024 / wrist 720×1280 | 8タスク: ブロック積み、ケーブル脱着、目薬をかご/棚へ、リング挿入、工具の挿入/取り外し | **人手の成功ラベル付き**（成功915 / 失敗1,532 / 準最適52）。ACT / Diffusion / SmolVLA / π0 / π0.5 / GR00T N1.7 / MolmoAct 2 のcheckpointが公開。Apache 2.0 |
| lerobot/svla_so101_pickplace | 50 | v3.0, 30fps | up / side 480×640 | SmolVLA公式のpick-place | 97個の派生モデルがHub上にある |
| 5hadytru/so101_bench_real_1_v2.1 | 3,203 | **v2.1**, 30fps | front / overhead 480×640 | 「物をかごへ」約1,200本＋指示追従約2,000本 | 手元環境で読める唯一の大規模データ。評価プロトコルは無し |
| youliangtan/so101-table-cleanup | 80 | **v2.1**, 30fps | front / wrist 480×640 | ペン・マーカー・テープをペン立てへ | 手元環境で読める |
| hbseong/record-pick-and-place-pos5-so101 | 240 | v3.0 | top / right 480×640 | 位置5点のpick-place | RQ1の位置条件と相性 |
| jackvial/so101_pickplace_success_120_v2 | 120（成功のみ） | v3.0 | top / side 600×800 | pick-place | 「成功のみ」で選別済み |

### 公開方策

- `lerobot/smolvla_base`（16.7万DL）: 入力はカメラ3台（256×256）＋state 6。欠損カメラは `empty_cameras` で埋める設計。
- `lerobot/pi05_base`（2万DL）: 現行lerobotのみ。カメラ3台 224×224、state 32次元にパディング。
- ArmnetBench の各タスク×各方策 checkpoint（例: `pravsels/act_block_stack_20k`）。**評価ロールアウトの動画と成功ラベルが公開されている**ので、比較対象の数字が既にある。

### ArmnetBench の公開結果（`armnetbench/reference_results.csv`、meta/episodes から算出）

成功率（成功本数/ロールアウト本数は CSV 参照。各方策×各タスク 30本、一部60本）:

| タスク | ACT | Diffusion | SmolVLA | π0 | π0.5 | GR00T N1.7 | MolmoAct 2 |
|---|---|---|---|---|---|---|---|
| 目薬をかごへ (eye_drops_to_basket) | **0.63** | 0.43 | 0.23 | **0.70** | **0.67** | **0.67** | 0.33 |
| ケーブルを外す (cable_unclip) | 0.60 | 0.07 | 0.37 | 0.47 | 0.70 | 0.33 | 0.28 |
| リング挿入 (ring_insert) | 0.50 | 0.23 | 0.38 | 0.17 | 0.47 | 0.60 | 0.10 |
| 目薬を棚へ (eye_drops_to_shelf) | 0.00 | 0.17 | 0.07 | 0.76 | 0.70 | 0.43 | 0.23 |
| 工具を外す (tool_removal) | 0.27 | 0.63 | 0.03 | 0.10 | 0.13 | 0.30 | 0.10 |
| 工具を挿す (tool_insert) | 0.12 | 0.23 | 0.03 | 0.40 | 0.53 | 0.17 | 0.08 |
| ブロック積み (block_stack) | 0.00 | 0.00 | 0.07 | 0.33 | 0.43 | 0.27 | 0.05 |
| ケーブルを挿す (cable_clip) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

判定は人手3値（成功/失敗/準最適）、成功のみを1と数える。参照デモは各タスク50本（テレオペ、全て成功扱い）、20fps。

### 参照デモの現場（eye_drops_to_basket、`armnetbench/frames/`）

- **物体**: 青い紙箱の目薬（"Eye Drops"の市販パッケージ）、透明プラスチックのかご（縦長・網目）、白い小さな台（棚: 折った板状）。
- **舞台**: 濃紺の WowRobo ロゴ入りマット（SO-101 キット付属のマットと同種）、周囲を段ボール箱で囲って背景を統一。
- **カメラ**: front はテーブル面の高さからアームの横を斜めに、top は斜め上から俯瞰、wrist はグリッパー先端を見下ろす。3台とも AV1 20fps、front/top 1024×576、wrist 1280×720。公開checkpoint（ACT）は3台を 640×480 で受ける。
- 開始: 目薬は台の上、かごはアームの左。タスクは台から目薬を取ってかごに入れる。

### 公開 checkpoint（README の表より）

| 方策 | eye_drops_to_basket | ring_insert |
|---|---|---|
| ACT | `pravsels/act_eyedrops_basket_20k` | `pravsels/act_ring_insert_20k` |
| Diffusion | `villekuosmanen/object_top_shelf_reset_remote_diffusion` | `villekuosmanen/armnetbench_ring_insert_diffusion` |
| SmolVLA | `pravsels/smolvla_eyedrops_basket` | `pravsels/smolvla_ring_insert` |
| π0 | `lorenzouttini/pi0-so101-object-top-shelf-reset-isambard-v50` | `lorenzouttini/pi0-so101-armnetbench-ring-insert-isambard-v50` |
| π0.5 | `lorenzouttini/pi05-so101-object-top-shelf-reset-isambard-v50` | `lorenzouttini/pi05-so101-armnetbench-ring-insert-isambard-v50` |
| GR00T N1.7 | `pravsels/groot1.7_eyedrops_basket_20k` | `pravsels/groot1.7_ring_insert_20k` |
| MolmoAct 2 | `pravsels/molmoact2_eyedrops_basket_20k` | `pravsels/molmoact2_ring_insert_20k` |

参照デモ: eye_drops_to_basket = `pravsels/object_top_shelf_reset_remote`、ring_insert = `villekuosmanen/armnetbench_ring_insert`（各50本、v3.0、20fps、Apache 2.0）。
lerobot 0.6.1 で直接動くのは ACT / Diffusion / SmolVLA / π0 / π0.5。GR00T と MolmoAct は別ランタイムが要るので第1段では外す。

### 課題の選定（案）

**主: eye_drops_to_basket。** 7方策中4つが0.63〜0.70で、落ち幅が測れる。物体は市販の目薬の箱・透明かご・白い台で、合計千円台。舞台のマットは手元のキット付属品と同種の可能性が高い（要確認）。
**副: ring_insert。** 木製ペグにリングを挿す玩具（数百円）で、精度の要る挿入課題として性格が違う。ACT 0.50 / GR00T 0.60。

避ける: cable_clip（全方策0%で再現しても情報なし）、block_stack（ACT/Diffusionが0%）。

### 機材の要件（ベンチマークの入力仕様＝ボトルネック特定済み）

公開checkpointは**3台のカメラ（front / top / wrist）を必須入力**にしており、欠損は埋められない（ACTは全入力が要る。黒画像で埋めるのは方策の入力分布を変えるので再現にならない）。手元は固定1台（C270）。**追加2台**が必要: 手首用（軽量・≦15g・USB、8/13の購入リストにあったもの）と top 用（C270級でよい）。USB帯域: 640×480×3台なら1コントローラで足りるが、ハブ経由は避ける。

## 2. 問いの候補（1つを主軸に選ぶ）

| ID | 問い | 何を測るか | 必要なもの |
|---|---|---|---|
| **RQ-P1 再現性** | 公開ベンチマーク（ArmnetBench）の公開checkpointは、**別個体・別部屋のSO-101**でどれだけ成功率を保つか | クロスリグ汎化。ArmnetBenchの自己申告成功率との差 | タスク物体（ブロック積み or 目薬＋かごが最安）、カメラ配置の再現（3台）、現行lerobot |
| RQ-P2 少数適応 | 公開データで学習した方策に自分のデモを 0/5/10/20 本足すと成功率はどう変わるか | 新しいリグに必要な「自前デモの量」（適応曲線） | RQ-P1の環境＋自前テレオペ |
| RQ-P3 品質 vs 量（公開版） | ArmnetBenchの成功ラベル付き2,499本で「成功エピソードのみ vs 全部」を学習し、自分のアームで評価 | RQ0と同じ問いを外部データで。RQ0の結果に一般性を与える | RQ-P1の環境（物体・カメラ共通）、GPU数晩 |

**推奨: RQ-P1 を主軸、RQ-P3 を第2実験。** 理由: (1) 第三者が定義し、成功ラベルと評価動画まで公開したベンチマークに対する**独立再現**は、この分野でほぼ誰もやっておらず「丁寧に測る人」の仕事そのもの (2) 比較対象の数字が既にあるので、n=20でも差の大きさが解釈できる (3) RQ-P3 は同じ物体・同じカメラ配置を使い回せる。

RQ-P2 は RQ-P1 で「落ちる」ことが確認できてから意味を持つ。

## 3. 決めること

1. **環境**: 新しい conda env（Python 3.12・lerobot 0.6.1・extras: core_scripts, training, smolvla, pi, feetech）を別名で作り、**現環境は RQ0/RQ1 のハーネス用に凍結**する。ハーネスの読み口（labbench の episode_parquet、score_rq1、command_vs_achieved、alignment_log）に v3.0 対応を足す必要がある。キャリブレーションファイルの形式が変わっている可能性があり、再キャリブレーションを見込む。
2. **カメラ台数**: ArmnetBench も smolvla_base も3台前提。手元は1台（C270）。追加2台は「ベンチマークの入力仕様」なので、購入規則（ボトルネック特定後）の要件を満たす。ただし**「1台だけでどれだけ落ちるか」も測定値として先に取れる**（欠損カメラは empty_cameras で埋まる）ので、購入前に1台での測定を挟む。
3. **タスク**: 8タスクのうち物体調達が最も安いのは block_stack（色付きブロック）と eye_drops_to_basket（目薬＋かご）。ArmnetBench 側の物体の寸法・色が README に無いので、参照デモ動画から読み取る。

## 3.5 環境構築の結果（2026-09-12 21:23〜21:40）

- `~/.venvs/lerobot2`（uv、Python 3.12.13、**lerobot 0.5.1**、torch 2.10.0+cu128、transformers 5.3.0、5.4GB）。
- **0.6.1 は Windows に入らない**: `core_scripts`/`dataset` extra が torchcodec≥0.7 を要求し、Windows用wheelが無い。uvは0.5.1に後退した。0.5.1はv3.0データセットを読み、processor pipeline（preprocessor/postprocessor同梱のcheckpoint）を扱え、act/diffusion/smolvla/pi0/pi05/groot/xvla を持つ。動画は torchcodec 無しで pyav に自動フォールバック（動作確認済み）。第1段はこれで進め、0.6.1が必要になった時点で torchcodec を外した手動インストールを検討する。
- **公開データ読み込み OK**: `pravsels/object_top_shelf_reset_remote`（50本、v3.0）をそのまま読めた（保存先 `~/.cache/huggingface/lerobot/hub/datasets--pravsels--object_top_shelf_reset_remote`）。
- **公開checkpoint推論 OK**（GPU、参照デモの1フレーム、教師状態）:

| checkpoint | 読込 | 1チャンク推論 | 予測 vs 正解（6関節） |
|---|---|---|---|
| `pravsels/act_eyedrops_basket_20k` | 26 s | **83 ms**（chunk 30 = 20fpsで1.5秒分） | [19.8, 1.0, -12.4, 70.5, 15.1, 38.2] vs [16.0, 2.3, -13.2, 70.2, 14.4, 37.3] |
| `pravsels/smolvla_eyedrops_basket` | 93 s | **428 ms**（chunk 50 = 2.5秒分） | [16.0, 1.8, -12.4, 69.4, 14.1, 35.5] vs 同上 |

  ACTは画像を 576×1024 / 720×1280 のまま受ける（設定の 480×640 は型記述で、resnet は任意サイズ）。SmolVLA は rename step で front/top/wrist → camera1-3 に写像し 256×256 へ。**手元のカメラ名も front / top / wrist に揃える**。
- **π0 / π0.5 / Diffusion の読込結果（同夜）**:
  - Diffusion `villekuosmanen/object_top_shelf_reset_remote_diffusion`: **0.5.1 では読めない**。front/top 576×1024 と wrist 720×1280 の混在を 0.5.1 の DiffusionPolicy が拒否する（「全画像同形状」の検査）。作成者はより新しい lerobot で学習している。0.6 系へ上げるか、その検査を外すパッチで対応可（次段）。
  - π0 / π0.5 `lorenzouttini/pi0(5)-so101-object-top-shelf-reset-isambard-v50`: **lerobot の checkpoint ではない**。openpi（JAX / orbax）の `step_24999/params` 形式で各12GB。lerobot には変換器が無く、openpi ランタイム（Linux 前提）が要る。第1段では除外し、そう明記する。
  - したがって第1段で本リグに載せられる公開方策は **ACT と SmolVLA の2つ**（Diffusion は環境更新後に追加）。
- **キャリブレーション**: 0.5.1 は `calibration/robots/so_follower/<id>.json`（旧 `so101_follower/`）を見る。旧ファイルをコピーして読込確認済み（形式は同一、6モーター）。リーダーも `teleoperators/so_leader/` へコピー済み。
- **カメラ**: 21:50 時点で index 0 が新旧どちらの環境からも開けない（別プロセスが掴んでいるか、USB の再接続が必要）。`align_camera.py watch` の窓が残っていないか確認する。
- `peft` は未導入（微調整時に要るなら追加）。

## 3.6 参照デモに対する offline 予測（`armnetbench/offline_reference_check.py`、2026-09-12 21:55）

50本×5フレーム刻み、教師状態。予測チャンク先頭と正解の平均絶対誤差（関節単位）と、グリッパー開閉の一致率:

| checkpoint | pan | lift | elbow | wrist_flex | wrist_roll | gripper | 開閉一致 | 所要 |
|---|---|---|---|---|---|---|---|---|
| ACT | 1.6 | 3.3 | 3.3 | 1.9 | 2.0 | 1.3 | 0.973 | 150 s |
| SmolVLA | 0.6 | 1.0 | 1.1 | 0.6 | 0.6 | 0.6 | 0.991 | 786 s |

両方とも自分の学習デモは offline で再現する（sanity floor 通過）。**SmolVLA の方が offline では正確なのに、公開の実機成功率は ACT 0.63 に対し 0.23。** offline 一致が実機を予言しないという、RQ0 の監査 D と同じ像がベンチマーク側でも出ている。RQ-P1 の結果を書くときの前置きになる。

## 4. 最初の一手（RQ0の60試行の後。実験中は重い処理を走らせない）

1. 新env構築（1〜2時間、ディスク約10GB）。動作確認は `lerobot-find-cameras` と v3.0 データセットの読み込みまで。
2. ArmnetBench の block_stack の参照デモ50本と ACT checkpoint を落とし、**GPUだけで**offline推論が通ることを確認（`offline_closure.py` の要領）。
3. 参照デモの動画からカメラ配置と物体を読み取り、調達リストを作る。
4. ここまでで RQ-P1 の事前登録文書を書く。評価は自分のアームで n=20/タスク、人手判定（ArmnetBench と同じ3値: 成功/失敗/準最適）。

## 5. やらないこと

- 自分の60本デモを使った新しい学習・実験（RQ0で閉じる）
- 現環境の lerobot をその場でアップグレードする（ハーネスが壊れる。別envにする）
- 物体・カメラの購入を測定より先にする
