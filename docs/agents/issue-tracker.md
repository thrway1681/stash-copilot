# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues on **`thrway1681/stash-copilot`**.
Use the `gh` CLI for all operations.

## ⚠️ Identity guard — read before any `gh` write

This repo is published under a **pseudonymous** GitHub account (`thrway1681`). This
machine's default GitHub identity is a separate **personal** account. `gh` operations
(creating issues, commenting) are attributed to whatever account `gh` is authenticated
as — so a misconfigured `gh` would **deanonymize the repo**.

**Before any `gh issue` / `gh label` / `gh pr` write, verify the active identity:**

```bash
gh api user --jq .login    # MUST print: thrway1681
```

If it prints anything else (e.g. the personal account), STOP — do not write. Re-auth as
the pseudonym first (`gh auth login` and sign in as `thrway1681`, or
`GH_TOKEN=<thrway1681-PAT> gh ...`). Read-only commands are lower risk but still prefer
the pseudonym token to avoid associating the accounts in GitHub's logs.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

Infer the repo from `git remote -v` — `gh` does this automatically when run inside a clone.

## When a skill says "publish to the issue tracker"

Create a GitHub issue (after the identity check above).

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.
