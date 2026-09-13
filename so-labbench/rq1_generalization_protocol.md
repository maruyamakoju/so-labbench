# RQ1: Generalization Benchmark v1 実験プロトコル

## v1.5 訂正（2026-09-13、RQ1 凍結中）

**不具合**: 秒からフレームへの変換が `round()` だった。Python の `round` は偶数丸めなので、
0.5 秒 = 7.45 フレーム（実測 14.9 Hz）は 7 に丸められ、**0.47 秒の閉じが「0.5 秒以上」として
通っていた**。同じ理由で境界の挙動が 29 Hz と 31 Hz で違っていた。`math.ceil` に変更。

**Pilot v1.3 の再採点結果への影響**: 報告されるフレーム閾値のみが変わった。

| | 変更前 | 変更後 |
|---|---|---|
| ACT 試行4,5 の `stable_f`（14.9 Hz） | 7 | **8** |
| SmolVLA 試行1-4 の `hold_f` | 19 | **20** |

**`success` / `grasp` / `lift` / `hold` / `taxonomy` は 1 件も変わっていない**（全10試行が
`grasp_failure`、成功 0）。閉じが一度も起きていないので、閾値が1フレーム動いても判定に届かない。
出力CSVに `n_closures` / `window_onset` / `lift_gain` を追加した。

**「scorer不具合→版を上げてPilotやり直し」規則との関係**: RQ1 は凍結中で、解凍時には基準方策が
変わっている（RQ0 のアームか公開 checkpoint）ため、いずれにせよ新しい Pilot が必要になる。
その Pilot は v1.5 で実施する。**判定に影響しなかったからといって規則を曲げたのではなく、
やり直すべき対象がまだ存在しないという意味である。**

## v1.4 修正（2026-09-12、RQ0評価前に適用。RQ1は凍結中）

**不具合**: `score_rq1.py`は最長の閉じ区間を把持とみなしていたが、学習デモ60本中27本が前エピソードの終了状態を引き継いで閉じたまま開始し、その待機区間が最長になる。把持と無関係な区間で持ち上げを測っていた（MORNING_REPORT_0816 判断2）。

**修正**: 安定長（0.5秒・実測レート換算）以上の閉じ区間のうち、**持ち上げ量が最大の区間**を把持窓とする。出力CSVに`n_closures`/`window_onset`/`lift_gain`を追加。

**検証**: 60デモに対し、把持+持ち上げの検出が旧18/44→23/60となり、品質監査（`so101_pick_remote_quality.csv`）の`complete_pickup` 23本と**60/60で一致**。Pilot 10本は閉じ皆無のため出力不変（回帰確認済み）。

**規則との関係**: v1.3のPilot 10本は本修正の影響を受けないので破棄しない。RQ1を再開する時点で基準方策が変わる可能性が高く（RQ0の結果次第）、その場合はいずれにせよ新しいPilotが要る。RQ1再開時のPilotはv1.4で行う。

## v1.3 修正（2026-08-15、Pilot実行中に発見・規則通りbump）

**発見した不具合**: `score_rq1.py`の成功判定が固定フレーム数（安定閉鎖15、保持30、コメントは"@30fps"）だったが、**実際の制御レートは30Hzに達していない**。lerobotの`record.py`は`while timestamp < control_time_s`で**壁時計**を回し、1周につき1フレーム記録する（`busy_wait(1/fps - dt_s)`は超過時に待たない）。データセットの`timestamp`列は`frame_index/fps`の合成値なので、この事実を隠す。

**実測**（Pilot、同一セッション・同一45秒・連続実行）:

| モデル | フレーム数 | 実効レート | 旧15フレーム | 旧30フレーム |
|---|---|---|---|---|
| ACT | 1063 | 23.6 Hz | 0.63秒 | 1.27秒 |
| SmolVLA | 830 | 18.4 Hz | 0.81秒 | 1.63秒 |

過去データも整合（ACT系1429-1441フレーム対SmolVLA系1122-1151、比1.27）。

**なぜ不具合か**: (1)プロトコルは秒で定義（0.5秒/1.0秒）しているのに実装は満たしていない (2)より重大: **重いモデルほど同じフレーム数が長い実時間を要求する**。SmolVLAはACTより1.28倍長い保持を課されていた。モデル間の劣化量を比較するベンチマークで、成功定義がモデル依存に偏るのは致命的。

**修正**: 閾値を秒で保持し、各試行の実測レート（フレーム数÷episode_sec、`rq1_manifest.csv`から取得）でフレーム数に変換する。出力CSVに`true_hz`/`stable_f`/`hold_f`/`hold_sec`を記録し監査可能にした。

**規則に従いPilotは全10本やり直し**（旧Pilot 2本 `eval_rq1_pilot_act_1` / `eval_rq1_pilot_smolvla_1` は破棄・混ぜない）。

## Pilot取り扱い規則（最終・2026-08-13 03:20）

Pilot（PILOT-001〜010）は**計測系の校正**であり本番データではない。Pilotでscorer不具合が見つかった場合の手順: **修正→v1.3にbump→Pilot全10本を最初からやり直し→10/10一致した版をfinal freeze**。古いPilotと新しいPilotは混ぜない。10/10達成後は**C0完了までsuccess定義・taxonomy・scorerを変更しない**。修正不要ならv1.2をfinal preregistration扱いとする。Pilotの成功率について一切考察しない（目的はmeasurement validityのみ）。求める唯一の出力: `Human label = Auto label 10/10`。

## v1.2 pre-pilot operational amendment（2026-08-13、データ取得前）

- **Taxonomy修正**（pick-upタスクとの整合）: SUCCESS=`success` / FAILURE=`approach_failure, grasp_failure, lift_failure, hold_failure, policy_instability` / EXCLUDED=`invalid_trial`（USB切断・録画失敗・対象誤配置・試行中の人的介入・ハード異常。policy failureに数えず再試行。`<prefix>invalid.txt`で採点から除外）。placementはRQ2用に予約、transportは廃止
- **Pilot gate**: 自動判定と目視判定の一致 **10/10を目標、1件でも不一致なら原因を特定してから本番**。本番C0でも最初の10-20試行は人間判定を併記しscorer driftを監査
- **Baseline gate（floor effect防止）**: Baseline成功率**≥70%で原則GO**。片方のモデルのみ著しく低い場合、そのモデルのshift比較は「探索的」と明記
- **結果の主表現はΔ（劣化量）**: Δ = Success_shift − Success_baseline。robustness profileを作る（モデルランキングをしない）
- **C5後の手順**: カメラを基準位置へ復元→baseline check 5試行で復元確認→C6
- **実験中の追加禁止**: 途中で思いついた実験は`future_experiments.md`に記録のみ（次Sprint検討）。新ハード・新モデル・RL・Isaac等はRQ1-A完了まで禁止
- Sprint終了の定義: Research Memo #001（Finding 3〜5個、各々にtable/graph/video evidence付き）の完成

## v1.1 事前登録修正（2026-08-13、データ取得前・以後変更禁止）

1. **Pilot 10試行を先行**（ACT5+SmolVLA5、本番データに不算入）。確認: 動画保存・scorer読取・自動判定vs人間判定の一致率≥95%・taxonomy曖昧性・カメラ/開始位置の再現性・1試行の実測時間。一致率不足ならscorer修正後に再Pilot
2. **モデル交互実行**: ACT→SmolVLA→SmolVLA→ACT…（ペア内順序はペア番号偶奇で反転）。開始位置P1-P5は両モデルで同一系列。時間・温度・照明ドリフトのconfound対策
3. **Daily baseline check**: シフト条件の実施前に毎日Baseline 5試行。前日比で大幅劣化なら装置ドリフトを疑い、条件データを取らない
4. **条件IDと実施順を確定**: C0 Baseline→C1 Position→C2 Appearance→C3 Distractor→C4 Lighting→C5 Camera→C6 Geometry（C5後はカメラを基準位置へ復元し、baseline checkで復元を確認してからC6）
5. **ログschema固定**（1 trial=1 row）: run_id, timestamp, model, condition, start_position, success, failure_type, grasp_success, lift_success, hold_success, inference_latency, human_intervention, video_path, notes。failure_typeは approach/grasp/transport/placement/policy_instability の5値固定、追加禁止
6. **動画は全trial保存**し `rq1/<condition>/<model>/trial_NNN.mp4` 構成で整理（元データはlerobotデータセット内、整理はコピー）
7. **n=20の解釈制限を明記**: 本ベンチマークは大きなfailure mode（>30pt差）の探索であり、微差の優劣判定はしない
8. **RQ1-B（第2週）の設計を差し替え**: Dataset H（60本・ほぼ同一環境）vs Dataset D（12本×5位置）を新規収録して比較（各60本、同一アーキ・同一学習設定）。余裕があればD+（位置+軽い照明・外観変動）。既存60デモのサブセット学習は予備解析としてのみ使用
9. RQ2への移行判定: RQ1で観測したfailureのうちCV/VLM Hybridで改善しうるものがある場合のみRQ2設計に進む

Sprint #001（2週間）/ 状態: 実行前 / 対応する研究メモ: `SO-ARM101 Generalization Study #001`

## Research Question

SO-ARM101のpick-up policyは、position / appearance / lighting / distractor / camera shiftの**どれに最も弱いか**。
また、demonstration数を増やすよりdemonstration diversityを増やす方がgeneralizationは改善するか。

## Hypothesis（事前登録）

- H1: camera shiftが最大の劣化を起こす（校正された視点への依存）
- H2: SmolVLA（事前学習済み視覚表現）はappearance/distractor shiftでACTより劣化が小さい
- H3: 同数のデモなら、diverse配置の方がhomogeneous配置より未知条件成功率が高い

## 使用資産（新規学習・新規ハード不要）

- Policy: `act_aug_v2_60ep` / `smolvla_v2_60ep`（v1凍結済みcheckpoint）
- タスク: リモコンのpick-up（60デモの学習タスクそのまま。**成功率を上げる活動はしない——測るだけ**）
- 注: 対象物はリモコン。学習時の配置・照明の再現が Baseline

## 実験マトリクス（一度に一変数）

| # | Condition | 変更 | 実施方法 | 他は全固定 |
|---|---|---|---|---|
| 0 | Baseline | なし | 学習時と同じ場所・照明・配置 | — |
| 1 | Position | 対象を学習分布外へ±5〜10cm | マット上に印を付けた3地点（近/遠/横） | ✓ |
| 2 | Appearance | 対象の色替え | 同じリモコンに青テープを巻く（形状不変） | ✓ |
| 3 | Lighting | 照明変更 | 部屋灯OFF+卓上ランプ（斜光） | ✓ |
| 4 | Distractor | 周囲に物体3個 | ペン・カップ・スマホを対象から10cm圏に | ✓ |
| 5 | Geometry | 対象の形状替え | 円柱物体（遠沈管到着後）or 同サイズの箱 | ✓ |
| 6 | Camera shift | カメラ移動 | 横に5cm平行移動（メジャーで計測・記録） | ✓ |

**実施順は0→1→2→3→4→5→6固定**（カメラ移動は他条件を汚染するため最後）。
各条件で **ACT 20試行 + SmolVLA 20試行**をシーン設置のまま連続実施（設営コスト共有）。

## 試行手順（全条件共通）

> **2026-09-13 の注記（本文は事前登録として凍結、以下は追記）**
> 本文が指す `run_trials.ps1` / `score_trials.py` は `harness/archive/` に移した。
> 現行は `run_interleaved.ps1`（Nアーム交互・マニフェスト記録）と `score_rq1.py`（v1.4）。
> RQ1 を解凍する際は現行ツールで実施し、手順の差分をここに追記すること。
> 採点の閾値・データセット配置・run_id の綴りは `harness/labbench.py` に集約された。
> 事前登録した仮説・条件・試行数・判定規則は一切変更していない。



1. `run_trials.ps1 -PolicyPath <policy> -RepoPrefix eval_rq1_<model>_<cond>_ -Trials 20 -EpisodeSec 45`
2. 毎試行前に自動起立（home_pose）、温度62°Cで自動クールダウン、接続脱落は自動リトライ
3. 人間の役割: 試行間に対象をその条件の規定位置へ戻す（条件1は3地点をローテーション）
4. 全試行の動画は自動保存（データセットとして残る）

## 成功定義と段階

pick-upタスクなので: **成功 = 安定把持（閉鎖0.5秒以上）+ 持ち上げ（閉鎖後lift上昇≥10）+ 保持1秒**
段階: approach（対象方向への下降）→ grasp（安定閉鎖）→ lift（把持したまま上昇）

## Failure Taxonomy（5分類、`score_rq1.py`が自動分類+動画で確認）

| 分類 | 定義 |
|---|---|
| approach_failure | 対象へ向かわない/下降しない |
| grasp_failure | 下降するが安定閉鎖なし |
| transport_failure(lift) | 閉鎖するが持ち上げられない/落とす |
| placement_failure | （本タスクではN/A、RQ2用に予約） |
| policy_instability | 無意味な動き・振動・停止 |

## Data Diversity実験（第2週）

同一条件（未知position）に対する成功率で比較：

| Dataset | 構成 | checkpoint |
|---|---|---|
| 30-mixed | 既存ep0-29（夜間・位置ばらつき小） | act_so101_pick_remote（既存・学習済み） |
| 60-mixed | 既存ep0-59（夜昼混合） | act_aug_v2_60ep（既存・学習済み） |
| 30-diverse vs 30-homog | 既存60デモからepisodesサブセット指定で再学習（新規収録なし、GPU夜間2本） | 新規学習2本 |

→ **More Data vs Better Distributed Data** の一次データ。
（注: 既存データの位置ラベルがないため、サブセット設計は動画から位置を目視分類してから確定する）

## 統計上の注意（正直に書く）

n=20/条件の95%信頼区間は最悪±22pt（p=0.5時）。本ベンチマークの目的は**条件間の順位付けと大きな差（>30pt）の検出**であり、5pt差の議論はしない。研究メモにはWilson CIを併記。

## 工数見積り（現実）

- 280試行 × 約2.7分 ≈ **12.6時間の実機稼働**
- 推奨ペース: **1日1条件**（40試行 ≈ 1.8時間、うち人間の拘束は対象戻しのみ）→ 7日で本表完了
- 第2週: diversity再学習（夜間無人）+ その評価（2条件×20×2 ≈ 3.6時間）+ 研究メモ執筆

## 成果物（Sprint終了時）

1. 研究メモ #001（1〜5p、規定形式）
2. Generalization matrix 1枚（条件×モデル×成功率+CI）
3. グラフ2〜3枚（段階別劣化・taxonomy分布・diversity比較）
4. 代表成功/失敗動画各3本
5. 外部レビュー1名

## 準備物（ユーザー、100均レベル）

- 青テープまたは青い紙（appearance用）／卓上ランプ（lighting用）／ペン・カップ等の distractor 3点／マットに位置マーカー（テープ印+メジャー写真で記録）
