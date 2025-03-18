# shared_codes_astro

Some code. Nice.


# Making a proper project out of it

1. Let's add uv. We wont do this as a workspace, and we'll use the flat structure.
   1. `uv init --app`
   2. Add our project dependencies: `uv add numpy scipy pandas matplotlib`
   3. Add dev dependencies: `uv add --dev ruff pre-commit`
   4. Set up some ruff defaults in the pyproject.toml, just the line length and target python should do for now.
2. Let's add a precommit. Copy one from the internet is easiest. I'll put one below.
3. Maybe a makefile so people don't have to remember what to run.
4. At this point I've only got the plot_powerspectrum_fors... okay lets rename this and see if we can clean the code. Don't forget the VSCode ruff extension.


```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: check-added-large-files
        args: ["--maxkb=5000"]
      - id: check-toml
      - id: check-json
      - id: check-symlinks
      - id: debug-statements
      - id: detect-private-key
      - id: check-yaml
        args:
          - --unsafe
      - id: trailing-whitespace
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.0
    hooks:
      - id: ruff
        args: ["--fix", "--no-unsafe-fixes"]
      - id: ruff-format
```
