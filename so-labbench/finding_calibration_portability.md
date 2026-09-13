# 所見: SO-101 の公開方策の行動値は、そのままでは他人のアームで意味が定まらない

2026-09-13 / 状態: **公開メタデータのみで示せる部分は確定。実機での帰結は RQ-P1 の門0で検証する。**
きっかけ: ArmnetBench の再現（`rqp1_armnetbench_reproduction_protocol.md`）の準備中に、
公開 checkpoint が我々のアームに送る指令値を確認したこと。

---

## 主張

LeRobot の SO-101 方策が出力する関節値は角度でも距離でもなく、**そのアームの較正レンジに対する
割合**である。較正レンジは `lerobot-calibrate` の実行時に操作者が各関節をどこまで動かしたかで
決まる。したがって、

1. 較正レンジが違う2台では、**同じ数字が別の物理位置**を指す
2. `drive_mode` が違えば、**同じ数字が逆向き**を指す
3. 学習側のレンジが受け側より狭ければ、方策は受け側の可動域の一部しか使えない
4. 学習側が広ければ、範囲外の指令は**黙って切り詰められる**

そして公開データを見る限り、**SO-101 のコミュニティはこの較正を共有していないし、公開もしていない。**

## 機構（`lerobot/motors/motors_bus.py`）

```python
bounded_val = min(max_, max(min_, val))          # 範囲外は黙って切り詰め
norm = ((bounded_val - min_) / (max_ - min_)) * 100
normalized_values[id_] = 100 - norm if drive_mode else norm     # drive_mode で反転
```

`min_` / `max_` は `calibration[motor].range_min` / `range_max`、つまり較正ファイルの値。
`_unnormalize` は同じ式の逆で、方策の出力を tick に戻すときに**受け側の較正**を使う。
学習時の較正と実行時の較正が違えば、その差はそのまま物理位置の差になる。

## 証拠1: 公開データセットは同じ尺度を使っていない

Hugging Face 上の SO-101 データセット 10 本について、`meta/stats.json` の行動値の範囲
（`armnetbench/public_gripper_ranges.csv`）:

| データセット | グリッパー最小 | 最大 | 使った幅 |
|---|---|---|---|
| hbseong/record-pick-and-place-pos5-so101 | 0.5 | **100.0** | 99.5 |
| 5hadytru/so101_bench_real_1_v2.1 | 0.0 | 91.0 | 91.0 |
| jackvial/so101_pickplace_success_120_v2 | 0.0 | 56.8 | 56.8 |
| villekuosmanen/armnetbench_block_stack | 0.0 | 52.3 | 52.3 |
| **pravsels/object_top_shelf_reset_remote**（我々の再現対象） | 0.0 | **48.4** | 48.4 |
| pravsels/cable_clip_remote_v2 | 0.0 | 45.5 | 45.5 |
| lerobot/svla_so101_pickplace | 0.0 | 33.0 | 33.0 |
| villekuosmanen/armnetbench_ring_insert | 0.0 | 29.5 | 29.5 |
| villekuosmanen/armnetbench_tool_insert | 0.2 | 28.4 | 28.1 |
| szk1ck/so101-ycb-pickplace | 0.05 | **1.5** | 1.4 |

参考: 我々自身の収録は 22.7〜100.0。

最後の1本は較正の違いではなく**尺度そのものが違う**（全関節が ±1.57 の範囲＝ラジアン）。
つまり公開 SO-101 データは、較正どころか**単位も共有していない**。

外れ値を除いても、グリッパーの最大指令値は **28.4 から 100.0 まで 3.5 倍の開き**がある。
これを「操作者の癖」と読むこともできるが、その解釈でも結論は変わらない——
**方策はその癖ごと学習し、他人のアームにその癖を出力する。**

## 証拠2: 解釈に必要な較正は公開されていない

較正ファイルさえあれば、受け側は指令値を物理位置に戻して再正規化できる。しかし:

| 公開物 | ファイル数 | 較正ファイル |
|---|---|---|
| `pravsels/object_top_shelf_reset_remote`（デモ50本） | 20 | なし |
| `villekuosmanen/armnetbench_block_stack` | 15 | なし |
| `armnet/armnetbench_v01_lerobot_so101`（ベンチマーク本体） | 633 | なし |
| `pravsels/act_eyedrops_basket_20k`（checkpoint） | 9 | なし |

較正 JSON は 1 kB 弱である。**これを同梱しないことで、公開された行動値は原理的に復元不能になる。**

## 我々のアームでの具体的な数字

我々の follower の較正: グリッパー 1400〜2462 tick（全可動域 1062 tick）。
再現対象の方策が出す指令をこの目盛りに載せると:

| 方策の指令 | 意味 | 我々のアームでの tick | 全開に対する割合 |
|---|---|---|---|
| 1.6 | 彼らの把持時の平均 | 1417 | 2% |
| 34.4 | 彼らの「開」の平均 | 1765 | 34% |
| 48.4 | 彼らの「開」の最大 | 1914 | 48% |
| 85.7 | **我々自身のデモの「開」の平均** | 2310 | 86% |

方策は我々のグリッパーに、我々のテレオペが使った開きの **1/3〜1/2 しか要求しない。**

## 何が示せていて、何が示せていないか

**示せている**（公開メタデータと lerobot のソースのみ）:
- 正規化は較正レンジ依存であり、`drive_mode` で反転しうる
- 公開 SO-101 データセットは尺度を共有していない（最大指令値に 3.5 倍の開き、うち1本は別単位）
- 較正ファイルを公開しているデータセット・checkpoint は、調べた範囲で1つもない

**示せていない（予測）**:
- この差が実機の成功率を実際に下げること。**RQ-P1 の門0**（`gripper_check.py` で開き幅を実測し、
  物体が通るか確認）と門3（人手デモのオープンループ再生）が最初の検証になる
- 較正を揃えれば直ること

**やってはいけない解釈**: これをもって「公開方策は使えない」と言うこと。示したのは
「**公開されている情報だけでは、行動値の物理的な意味が決まらない**」であって、
方策の良し悪しではない。

## 単独で完結する追試（アーム1台・1時間）

1. グリッパーを全可動域で較正 → 指令 40 の開き幅を定規で測る
2. 意図的に半分のレンジで再較正 → 同じ指令 40 の開き幅を測る
3. 同じ数字が別の物理位置になることを直接示す

`rqp1_design_review.md` の筋3。RQ-P1 の門0で 1 は測るので、2 を足すだけで済む。

## 提案（公開する側へ）

データセットと checkpoint に **`calibration.json` を同梱する**。1 kB で、行動値が復元可能になる。
それが無い限り、「この SO-101 の方策をあなたの SO-101 で動かせます」という主張は、
共有されていない前提に依存している。

## 出典

- 正規化の実装: `lerobot/motors/motors_bus.py` の `_normalize` / `_unnormalize`（lerobot 0.5.1）
- 各データセットの `meta/stats.json` と `meta/info.json`（2026-09-13 取得、`public_gripper_ranges.csv`）
- 我々の較正: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json`
