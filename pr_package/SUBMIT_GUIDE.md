# PR提出手順（10分・要あなたのGitHubアカウント）

## 前提
- GitHubアカウントでログイン済みのブラウザ、または `gh auth login` 済みのターミナル

## 手順（コマンド版）
```
# 1. Fork (ブラウザで https://github.com/huggingface/lerobot → Fork)
# 2. clone & branch
git clone https://github.com/<あなたのID>/lerobot
cd lerobot
git checkout -b fix-record-loop-infinite-reset

# 3. パッチ適用
git apply <このフォルダ>/fix_infinite_reset_loop.patch

# 4. commit & push
git add src/lerobot/scripts/lerobot_record.py
git commit -m "Fix infinite reset loop in record_loop when no teleoperator is provided"
git push -u origin fix-record-loop-infinite-reset

# 5. ブラウザでPR作成。本文は PR_BODY.md をコピペ
```

## 先にIssueを立てる場合
ISSUE.md の内容を https://github.com/huggingface/lerobot/issues/new に貼る。
PR本文に "Fixes #<issue番号>" を追記。

## 注意
- 提出前に同種のIssue/PRが既にないか検索: "record_loop reset" "infinite loop" "skipping action generation"
- CONTRIBUTING.mdに従いpre-commit要求がある場合: `pip install pre-commit && pre-commit run --files src/lerobot/scripts/lerobot_record.py`
