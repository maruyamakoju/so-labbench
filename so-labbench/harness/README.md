# SO-LabBench Harness 使い方（1ページ）

すべて `LeRobot Prompt.bat` のコンソール、またはフルパスのenv python（`labbench_config.json`参照）で実行。

## RQ1の1条件を回す標準フロー

```powershell
# 1) セッション開始: カメラを基準位置へ（下の「カメラ位置」節。合格は必須ではない）
python align_camera.py watch
powershell -File align_view.ps1

# 2) 2モデル交互にN ペア（=2N試行）。起立→実行→温度ゲート→リトライは自動
powershell -File run_interleaved.ps1 -Condition c0 -Pairs 20 -EpisodeSec 45 -ResetSec 20

# 3) 後処理ひとまとめ（採点・実効レート・カメラ残差・フィルムストリップ）
powershell -File finish_condition.ps1 -Condition c0 -Pairs 20
```

Pilotだけは追加で、目視判定を`pilot_human_labels.md`に記入してから：

```powershell
python pilot_agreement.py      # 人間判定 vs 自動判定。ゲートは10/10
```

**使われなくなった道具は `archive/` に移した**（理由は `archive/README.md`）。特に
`score_trials.py` は現行 `score_rq1.py` と同じ CSV 名に**異なる定義**で上書きするので、
戻して走らせてはいけない。

```powershell
python test_harness.py          # 計測コードの回帰テスト（156件、pytest不要）
python test_harness.py pipeline # 全工程の統合テスト（合成した20試行を採点→盲検→集計まで通す）
python mutation_check.py        # そのテストが本当に効いているか（15個のバグを注入して全部検出されるか）
python labbench.py              # いま何を見ている設定か（パス・カメラ・閾値・安全値）
```

**テストが通ることは、それ自体では何の証拠でもない。** コードの後に書いたテストは「あるべき
挙動」ではなく「いまの挙動」をなぞりがちなので、`mutation_check.py` で既知のバグを19種入れ直し、
全部が検出されることを確かめる。生き残った変異は、そこにテストの穴があるという意味。
実際これで2つ見つかった（14.9Hz での切り上げ境界と、盲検の並び替えが効いていない件）。

## RQ0: データ品質仮説（2026-09-12〜、`../rq0_data_quality_protocol.md`）

RQ1は基準条件で動く方策がないため凍結。先に「不完全なデモを除くと閉じを指令するようになるか」を測る。

```powershell
python rq0_subsets.py                      # 品質CSVから3アームのエピソード一覧を導出 -> rq0_subsets.json
powershell -File train_rq0.ps1             # 23complete, 23random を順に学習（ACT・100k・凍結checkpointと同設定・一晩）
powershell -File train_rq0.ps1 -Arms 23complete -Steps 10 -SaveFreq 10 -Tag _smoke   # 動作確認だけ（後で outputs\train\*_smoke を消す）
python command_vs_achieved.py              # 主指標: 指令グリッパー<40が実測レート換算0.5秒以上あった試行の割合 -> command_vs_achieved.csv
python offline_closure.py [stride]         # 探索的: 3アームが学習デモに対して閉じを予測するか（GPU・約20分） -> rq0_offline_closure.csv
```

学習には lerobot 本体へのパッチが必要（`../../pr_package/fix_episode_subset_indexing.patch`）: `--dataset.episodes` でサブセットを渡すと
`__getitem__` が元のエピソード番号でサブセット内の位置表を引いて `IndexError` になる。上流の現行版はデータセット層を書き直しており対象外。
評価は同じランナーで、アームを3つ渡す（`-DryRun` で順序だけ確認できる）:

```powershell
powershell -File run_interleaved.ps1 -Study rq0 -Condition base -Pairs 20 -Arms 60all,23complete,23random
powershell -File finish_condition.ps1 -Study rq0 -Condition base -Pairs 20 -Models 60all,23complete,23random
```

## RQ-P1: ArmnetBench の独立再現（2026-09-12〜、`../rqp1_armnetbench_reproduction_protocol.md`）

別環境（`~/.venvs/lerobot2`、lerobot 0.5.1）・3カメラ・20fps・ベンチマークの指示文で走らせる。**設定ファイルを切り替えるだけ**で同じスクリプトが使える。

```powershell
$env:LABBENCH_CONFIG = "$PWD\labbench_config_rqp1.json"     # harness ディレクトリで実行
$env:LABBENCH_CAMERA = "front"          # 残差ログ・フィルムストリップに使うカメラ

# 0) カメラの index を確認して labbench_config_rqp1.json の index_or_path を直す
& "$(python labbench.py --get paths.lerobot_scripts)\lerobot-find-cameras.exe" opencv

# 0.5) 事前チェック（env・カメラ3台のモード・較正・モーター・方策キャッシュ・基準画像・ディスク）と前カメラの露光固定
powershell -File preflight_rqp1.ps1
python lock_exposure.py --measure ; python lock_exposure.py --set -6      # LABBENCH_CAMERA=front の C270 に効く

# 1) 3台を順に参照デモの画角へ（基準画像は ../armnetbench/frames/reference_<cam>_ep0_f0.png）
$env:ALIGN_CAM_INDEX = "0"; python align_camera.py watch ..\armnetbench\frames\reference_front_ep0_f0.png
$env:ALIGN_CAM_INDEX = "1"; python align_camera.py watch ..\armnetbench\frames\reference_top_ep0_f0.png
$env:ALIGN_CAM_INDEX = "2"; python align_camera.py watch ..\armnetbench\frames\reference_wrist_ep0_f0.png
#    （別窓で powershell -File align_view.ps1 を開いて指示を見る。合格は必須ではない）

# 2) 順序確認 → 本番（各方策30回 = 60試行）
powershell -File run_interleaved.ps1 -Study rqp1 -Condition eyedrops -Pairs 30 -Arms act,smolvla -EpisodeSec 45 -ResetSec 20 -DryRun
powershell -File run_interleaved.ps1 -Study rqp1 -Condition eyedrops -Pairs 30 -Arms act,smolvla -EpisodeSec 45 -ResetSec 20

# 3) 後処理（v3.0 形式の録画も labbench.py が自動で読み分ける）
powershell -File finish_condition.ps1 -Study rqp1 -Condition eyedrops -Pairs 30 -Models act,smolvla
```

方策は Hub の id（`pravsels/act_eyedrops_basket_20k` / `pravsels/smolvla_eyedrops_basket`）を `--policy.path` に渡すので、ダウンロードは初回に自動。
`python align_camera.py` は旧環境の python（ヘッドレス cv2）で動く。参照画像が 16:9 なら照合も 16:9 で行う。

## 採点の時間定義（v1.3・重要）

lerobotの`record.py`は`episode_time_s`ぶんの**壁時計**を回し1周1フレーム記録する。
推論が間に合わなければフレーム数が減るだけで、**30Hzは出ていない**（実測 ACT 23.6Hz /
SmolVLA 18.4Hz）。データセットの`timestamp`列は`frame_index/fps`の合成値なのでこれを隠す。

したがって成功判定の閾値は**秒で保持し、各試行の実測レートでフレーム数へ変換**する
（`score_rq1.py`）。固定フレーム数にすると、**遅いモデルほど長い保持を要求される**
——モデル比較が目的のベンチマークでは致命的。`policy_instability`の反転指数も同じ理由で
毎秒あたりに正規化してある。出力CSVの`true_hz`/`stable_f`/`hold_f`/`hold_sec`/`rev_per_s`で検算できる。

```powershell
python trial_timing.py         # 各試行の実効レートと、閾値が何秒に相当するか
```

## カメラ位置（C0のbaselineとC5の復元が依存する）

学習データは**カメラ位置が2グループある**（ep0-29とep30-59、各グループ内は8px以内）。
RQ1のbaselineは**ep30-59側**を採用する——把持動作を教えたのがこの30本で、v2監査
(`eval_actv2_*`)もこの位置で撮られている（12px以内）。基準画像は`_frames/camera_reference.png`（ep45）。

```powershell
# セッション開始時: 合わせながら見る（左上に指示がライブ表示される）
python align_camera.py watch
powershell -File align_view.ps1        # 別窓。閉じると撮影ループも止まる

# 単発でズレを数値で見る
python camera_pose_check.py            # --json で機械可読

# 収録後: 各試行が実際どの位置で撮られたかを記録（カメラ不要・事後でOK）
python alignment_log.py                # rq1_manifest.csv の全runs -> rq1_alignment.csv

# データセットが単一のカメラ位置で撮られたかの監査
python dataset_pose_audit.py so101_pick_remote 3
```

**判定は門ではない。** ズレが許容外でも試行は止めない——`alignment_log.py`が全試行の
残差を残すので、後から「結果がカメラのドリフトで説明できるか」を確認できる。
許容値は環境変数で変えられる: `ALIGN_TOL_PX`(既定20) / `ALIGN_TOL_SCALE`(0.05) /
`ALIGN_TOL_ROT`(2.5) / `ALIGN_C5_SHIFT_PX`(60)。残差は**C5の摂動量に対する比**でも
出るので、`0.2×`なら測ろうとしている効果の1/5、という読み方ができる。

## 補助ツール

| ツール | 用途 |
|---|---|
| `home_pose.py` | アームを中央ポーズへ（終了時に速度制限を自動復元） |
| `health_motors.py` | 温度・電圧・エラーフラグ（読めない場合は `temp=UNREADABLE`。ランナーはこれを「冷えている」と読まず停止する） |
| `blind_review.py prepare/check` | 盲検レビュー一式（動画を`R001.mp4`等に振り直し、対応表は別ファイル） |
| `test_harness.py` | 計測コードの回帰テスト |
| `id_reader.py generate` | ArUcoマーカー印刷シート生成（ラック・キャップ識別用） |
| `id_reader.py detect [画像]` | マーカー検出（カメラ or 画像ファイル） |
| `make_filmstrip.py <prefix> <n>` | 試行ごとのコマ送り（**採点器の数値を載せない**＝人間判定を汚染しない） |
| `make_reference_frame.py <dataset> <ep...>` | 学習フレームから`_frames/camera_reference.png`を作る |
| `dataset_pose_audit.py <dataset> [step]` | データセットのカメラ位置をグループ分け |

## 運用ルール（監査で確立）

- 評価は**毎試行home_poseから**（開始姿勢が揃わないと方策が固まる）
- 接続時のモーター脱落は自動リトライ／モーター温度が閾値以上で自動クールダウン（回数と温度は config の `safety`）
- グリッパー規約: **100=開, 0=閉**、把持保持の実測値は〜25
- 判断は**lossではなく独立試行の成功率**で行う
- 中断された収録のデータセットdirは削除してから再実行（FileExistsError対策）
- **邪魔な変数は門で止めず記録する**。カメラ位置も実効レートも、試行を止める条件ではなく
  試行に付ける値として残す。門はデータを捨てるが、記録は後から交絡を検証させる
- **この機械の事実を定数に埋めない**。制御レート・カメラ位置・キャリブレーション依存の
  閾値は測って導出する。埋めると他人の機械で静かに意味が変わる（＝再現性が壊れる）
- **決定は1箇所にしか書かない**。閾値・データセットのパス・run_id の綴りは `labbench.py`
  にあり、他のスクリプトは定義しない。かつて同じ定数が8箇所にあり、うち1組は値が食い違って
  いた（品質監査は12フレーム、採点器は0.5秒＝15フレーム）。RQ0のアーム選択に実害は無かったが
  （60本中0本が判定変化、検証済み）、防ぐ仕組みが無かった
- **出力は「何が作ったか」で名前を付ける**（`<study>_<condition>_<name>`）。以前は研究を
  跨いで同じファイル名に書き込んでいた

## 未較正のまま残っている定数（Pilotの人間判定で決める）

`CLOSE_T=40`（把持とみなすグリッパー値）/ `LIFT_GAIN=10` / `DESCEND_D=15` /
`MOVE_D=25` / `INSTABILITY_REV_PER_S=2.7`。いずれも人間判定と突き合わせた較正を経ていない。
特に反転指数は実測が7-12/秒で、あの分岐に来た試行はほぼ全部`policy_instability`になる。
**推測で調整せず、Pilotの不一致として検出させる**方針。

これらは `labbench_config.json` の `scoring` ブロックにあり、環境変数でも上書きできる。
絶対パス・シリアルポート・ffmpeg パスの直書きは解消済み（`labbench.py` 経由、`LABBENCH_CONFIG` で切替）。
