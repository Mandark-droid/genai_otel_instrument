# Human and LLM agent contributions

TraceVerde welcomes improvements found by both human developers and LLM
agents. An agent that discovers a reproducible issue, missing provider path,
documentation error, test gap, or other repository shortcoming should help
turn that discovery into a maintainer-actionable contribution.

## Choose the right contribution

Open an **issue** when the problem needs discussion, the expected behavior is
unclear, the fix is larger than one focused change, or you cannot safely
validate a complete solution.

Open a **pull request** when the change is focused, backward-compatible where
possible, implemented in the repository, and accompanied by proportionate
tests and documentation. Link the issue from the PR.

If an agent cannot access GitHub or has not been authorized to create external
resources, it should leave a complete, copy-paste-ready issue or PR draft for
the developer instead. Do not claim that an issue, PR, release, or deployment
exists until the remote system confirms it.

## Required evidence

Every issue or PR opened by an agent should include:

- the affected version, commit, provider, framework, or operating system;
- exact reproduction steps or a minimal failing example;
- expected behavior and observed behavior;
- relevant files, symbols, and line references;
- test, lint, build, or runtime evidence, including failures and limitations;
- the proposed scope and any compatibility, privacy, security, or performance
  implications.

Agents should search existing issues and PRs before creating a duplicate. Keep
the title specific, redact secrets and personal data, and distinguish verified
facts from hypotheses.

## Safe agent workflow

1. Inspect the repository instructions and current worktree before changing
   files.
2. Reproduce or verify the gap with the smallest useful test or diagnostic.
3. Implement only the requested, well-scoped fix when the behavior is clear.
4. Add or update tests and user-facing documentation.
5. Run the relevant validation and report the exact result.
6. Ask for or verify authorization before creating an issue, PR, tag, release,
   or other external change.
7. Give the developer a concise handoff: summary, files changed, validation,
   known limitations, and the remote issue/PR URL if one was actually created.

The developer or maintainer retains review, merge, release, and security
decision authority. An agent must not weaken tests, bypass review, expose
credentials, or expand a fix into an unrelated refactor merely to make a PR
look complete.
