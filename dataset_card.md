# Dataset Card: so101_pick_remote

公開時はHuggingFace Hubの `README.md`（dataset card）としてそのまま使用可。

## 概要
- **内容**: SO-ARM101（so101_follower）による「机上のテレビリモコンを掴んで持ち上げる」遠隔操作デモンストレーション
- **規模**: 60エピソード / 43,352フレーム / 30fps / 約480MB（+把持重点の追加収録分）
- **形式**: LeRobotDataset v2.x（parquet + SVT-AV1動画）
- **収録**: 2026-08-11〜12、単一の家庭内デスク環境

## 構成
| 項目 | 内容 |
|---|---|
| observation.state | 6関節の正規化位置（pan, lift, elbow, wrist_flex, wrist_roll, gripper） |
| observation.images.fixed | 固定カメラ（Logicool C270, 640×480@30fps、側方視点） |
| action | Leaderアームからの6関節目標値（gripper: 100=開, 0=閉） |
| task | "Pick up the remote control" |

## バッチ構成（重要）
- **ep 0-29**: 夜間・室内照明。通常のpick-up操作
- **ep 30-59**: 昼光。把持の瞬間を低速・丁寧に演じた把持重点デモ
- 既知の特性: 31.0%のフレームがグリッパー閉状態 / 一部エピソードは閉状態で開始（直前エピソード終端の引き継ぎ）/ ep33,34,40,47,57は閉じ動作なし

## 収録条件
- Leader/Followerともキャリブレーション済み（キャリブレーションファイル同梱）
- カメラ位置は全収録で固定（基準構図画像あり）
- 収録者1名（右利き）、リモコン位置は毎エピソード数cm〜10cm変動

## 学習実績（参考）
- ACT（100k steps, image aug）とSmolVLA（smolvla_baseから30k steps）で学習可能なことを確認
- 実機評価では両者とも標的接近まで成功、把持発火はACT 2/5・SmolVLA 0/6（詳細: audit_report.md）
- 本データセットは「単一固定視点での模倣学習の限界検証」のベースラインとして利用可能

## 制限・注意
- 単一環境・単一操作者・単一物体であり、汎化性能の学習には不適
- 手首カメラなし（固定視点のみ）
- ライセンス: （公開時に選択: apache-2.0推奨）
