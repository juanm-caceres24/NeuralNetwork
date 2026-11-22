# Git Workflow Guide

This repository uses a **Simplified Solo Git Flow** tailored for a single-developer project without Pull Requests, balancing a clean history with safe, structured releases.

---

## 1. Branch Model

| Branch | Purpose | Rules |
| :--- | :--- | :--- |
| `main` | Stable, tested milestones/releases. | **Never commit directly.** Only receives merges from `development` (releases) or `hotfix/*`. Tagged as `vX.Y.Z`. |
| `development` | Active integration branch. | Must always compile and pass `./scripts/test.sh`. |
| `feat/<topic>` | New features or modules (e.g., `feat/dense-layer`, `feat/http-server`). | Branched from `development`; merged back to `development` and deleted once complete. |
| `fix/<topic>` | Bug fixes during development (e.g., `fix/softmax-overflow`). | Branched from `development`; merged back and deleted. |
| `perf/<topic>` | Performance optimizations (e.g., `perf/thread-pool`). | Branched from `development`; merged back and deleted. |
| `exp/<topic>` | Throwaway ML experiments (architectures, hyperparameters). | Branched from `development`; delete freely if discarded. |
| `hotfix/<topic>` | Urgent fix on a released version in `main`. | Branched from `main`; merged back into both `main` and `development`. |

---

## 2. Commit Message Convention (Conventional Commits)

Format:

```text
<type>(<scope>): <short imperative description>
```

### Common Types
- `feat`: New functionality (`feat(core): add DenseLayer forward pass`)
- `fix`: Bug fix (`fix(activation): prevent exp overflow in sigmoid`)
- `test`: Adding or updating tests (`test(core): add numerical derivative checks`)
- `docs`: Documentation changes (`docs: add project specification and architecture`)
- `perf`: Performance improvement (`perf(matrix): optimize matvec_transpose cache order`)
- `refactor`: Code restructuring without changing behavior
- `build`: Makefile, scripts, or dependency changes (`build: add debug and release targets`)
- `chore`: Maintenance tasks (`.gitignore`, cleanup)

---

## 3. Standard Commands

### Start a New Feature
```bash
git checkout development
git checkout -b feat/my-feature
```

### Commit Incrementally
```bash
./scripts/test.sh
git status
git add <files>
git commit -m "feat(core): implement feature summary"
```

### Merge a Feature into `development`
Use `--no-ff` to preserve the feature grouping in history (easy to inspect or revert as a single unit):
```bash
git checkout development
git merge --no-ff feat/my-feature -m "merge: integrate feat/my-feature into development"
git branch -d feat/my-feature
```
> **Tip:** If a feature branch accumulated messy WIP commits, use `git merge --squash feat/my-feature` followed by a single clean commit instead.

### Cut a Release to `main`
Once a milestone in `development` is complete and all tests pass in both `debug` and `release` modes:
```bash
./scripts/test.sh debug
./scripts/test.sh release

git checkout main
git merge --no-ff development -m "release: v0.1.0 core math and network forward pass"
git tag -a v0.1.0 -m "v0.1.0: core math and forward pass"
git checkout development
```

### Push to Remote
```bash
git push origin development
git push origin main --tags
```

---

## 4. Repository Hygiene Rules
- Never commit `build/`, compiled binaries, or object files (enforced by `.gitignore`).
- Avoid committing large datasets or multi-megabyte trained weight files; keep small example configs and synthetic generators versioned instead.
- Update relevant files in `docs/` and unit tests in `tests/` inside the same feature branch as the code they document/verify.
