# RQ-P1 人手判定シート（eye_drops_to_basket）

判定は ArmnetBench と同じ3値: `successful`（目薬がかごの中に入って終わる）/ `suboptimal`（入ったが乱暴・落下後に入った・かごが動いた等）/ `failure`。
試行として無効（USB切断・録画失敗・人的介入・誤配置）は `invalid`。**方策名を見ずに動画（フィルムストリップ）だけで判定する。**
順序は run_interleaved.ps1 の巡回規則と同じ（ラウンド k: (k-1) mod 2 だけ回転、位置 P((k-1) mod 5 + 1)）。

| run_id | round | position | policy | label | notes |
|---|---|---|---|---|---|
| eval_rqp1_eyedrops_act_1 | 1 | P1 | act | | |
| eval_rqp1_eyedrops_smolvla_1 | 1 | P1 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_2 | 2 | P2 | smolvla | | |
| eval_rqp1_eyedrops_act_2 | 2 | P2 | act | | |
| eval_rqp1_eyedrops_act_3 | 3 | P3 | act | | |
| eval_rqp1_eyedrops_smolvla_3 | 3 | P3 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_4 | 4 | P4 | smolvla | | |
| eval_rqp1_eyedrops_act_4 | 4 | P4 | act | | |
| eval_rqp1_eyedrops_act_5 | 5 | P5 | act | | |
| eval_rqp1_eyedrops_smolvla_5 | 5 | P5 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_6 | 6 | P1 | smolvla | | |
| eval_rqp1_eyedrops_act_6 | 6 | P1 | act | | |
| eval_rqp1_eyedrops_act_7 | 7 | P2 | act | | |
| eval_rqp1_eyedrops_smolvla_7 | 7 | P2 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_8 | 8 | P3 | smolvla | | |
| eval_rqp1_eyedrops_act_8 | 8 | P3 | act | | |
| eval_rqp1_eyedrops_act_9 | 9 | P4 | act | | |
| eval_rqp1_eyedrops_smolvla_9 | 9 | P4 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_10 | 10 | P5 | smolvla | | |
| eval_rqp1_eyedrops_act_10 | 10 | P5 | act | | |
| eval_rqp1_eyedrops_act_11 | 11 | P1 | act | | |
| eval_rqp1_eyedrops_smolvla_11 | 11 | P1 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_12 | 12 | P2 | smolvla | | |
| eval_rqp1_eyedrops_act_12 | 12 | P2 | act | | |
| eval_rqp1_eyedrops_act_13 | 13 | P3 | act | | |
| eval_rqp1_eyedrops_smolvla_13 | 13 | P3 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_14 | 14 | P4 | smolvla | | |
| eval_rqp1_eyedrops_act_14 | 14 | P4 | act | | |
| eval_rqp1_eyedrops_act_15 | 15 | P5 | act | | |
| eval_rqp1_eyedrops_smolvla_15 | 15 | P5 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_16 | 16 | P1 | smolvla | | |
| eval_rqp1_eyedrops_act_16 | 16 | P1 | act | | |
| eval_rqp1_eyedrops_act_17 | 17 | P2 | act | | |
| eval_rqp1_eyedrops_smolvla_17 | 17 | P2 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_18 | 18 | P3 | smolvla | | |
| eval_rqp1_eyedrops_act_18 | 18 | P3 | act | | |
| eval_rqp1_eyedrops_act_19 | 19 | P4 | act | | |
| eval_rqp1_eyedrops_smolvla_19 | 19 | P4 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_20 | 20 | P5 | smolvla | | |
| eval_rqp1_eyedrops_act_20 | 20 | P5 | act | | |
| eval_rqp1_eyedrops_act_21 | 21 | P1 | act | | |
| eval_rqp1_eyedrops_smolvla_21 | 21 | P1 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_22 | 22 | P2 | smolvla | | |
| eval_rqp1_eyedrops_act_22 | 22 | P2 | act | | |
| eval_rqp1_eyedrops_act_23 | 23 | P3 | act | | |
| eval_rqp1_eyedrops_smolvla_23 | 23 | P3 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_24 | 24 | P4 | smolvla | | |
| eval_rqp1_eyedrops_act_24 | 24 | P4 | act | | |
| eval_rqp1_eyedrops_act_25 | 25 | P5 | act | | |
| eval_rqp1_eyedrops_smolvla_25 | 25 | P5 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_26 | 26 | P1 | smolvla | | |
| eval_rqp1_eyedrops_act_26 | 26 | P1 | act | | |
| eval_rqp1_eyedrops_act_27 | 27 | P2 | act | | |
| eval_rqp1_eyedrops_smolvla_27 | 27 | P2 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_28 | 28 | P3 | smolvla | | |
| eval_rqp1_eyedrops_act_28 | 28 | P3 | act | | |
| eval_rqp1_eyedrops_act_29 | 29 | P4 | act | | |
| eval_rqp1_eyedrops_smolvla_29 | 29 | P4 | smolvla | | |
| eval_rqp1_eyedrops_smolvla_30 | 30 | P5 | smolvla | | |
| eval_rqp1_eyedrops_act_30 | 30 | P5 | act | | |
