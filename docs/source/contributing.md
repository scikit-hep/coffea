# Contributing to coffea

coffea is a community-driven project. This page highlights the most common
ways to get involved and the expectations we have for contributors of every
experience level.

## Ways to contribute

- Report reproducible bugs and help triage existing issues.
- Propose and discuss new features or major changes.
- Improve documentation, tutorials, and examples.
- Share feedback, success stories, or usage questions in Discussions or Slack.

## Getting set up

1. Follow the developer installation steps in the
   {doc}`getting_started/installation` guide to create a local environment.
2. Fork the repository on GitHub and clone it locally.
3. Install the development dependencies and activate the virtual environment
   you created in the previous step.

If you are new to contributing on GitHub, the
[GitHub quickstart guide](https://docs.github.com/en/get-started/git-basics/set-up-git)
walks through the basics of configuring git, forking, and cloning a project.

## Reporting issues and asking questions

- Search existing [GitHub issues](https://github.com/scikit-hep/coffea/issues)
  and [Discussions](https://github.com/scikit-hep/coffea/discussions) before
  opening something new.
- When filing a bug, include the coffea version, your Python version, and the
  minimal code needed to reproduce the problem.
- Feature requests are best discussed in a new issue or in Discussions. Provide
  the motivation, any existing workarounds, and how you would measure success.
- Usage questions are welcome in the
  [IRIS-HEP Slack `#coffea` channel](https://iris-hep.slack.com/)
  (invite instructions are pinned in the channel topic).

## Submitting changes

- Keep pull requests focused. Small, reviewable patches are merged faster.
- Reference any related issues in the description and explain the motivation,
  approach, and limitations.
- Update or add tests whenever you change behavior or add new features.
- Include relevant documentation updates so the docs stay in sync with the
  code.
- Run the automated checks locally before pushing:

  ```bash
  pre-commit run --all-files
  pytest
  ```

We prefer contributions that maintain or improve the existing code coverage.
If you run into trouble getting tests to pass, open a draft pull request and
ask for help.

## Documentation contributions

- Source files live in `docs/source`. Edit Markdown (MyST) or notebook files as
  needed.
- Build the documentation locally with:

  ```bash
  pushd docs
  make -j html
  # the `-j` option requests to use all of the processors on your machine
  # to make the build faster, you can limit to N processors using -jN or
  # one processor by omitting the `-j` option
  popd
  ```

- Preview the generated HTML at `docs/build/html/index.html` before opening
  your pull request.
- Parts of the documentation reference optional sub-components of coffea,
  so you may want to install all of the optional components in order to
  mimic the complete documentation site. `pip install -e .[dev,caches,dask,dask-awkward,rucio]`
- Additionally, parts of the documentation use `graphviz` to construct
  class inheritance trees. If you want to view those, install the `graphviz`
  system package as well (e.g. `sudo apt install graphviz` on Debian variants).
- The `make watch` command uses `sphinx-autobuild` to re-build the pages as
  you change the source files which can be helpful depending on your workflow.
  It spawns a simple local HTTP server as well which reloads the site after
  the re-build is completed.

## Release cadence

coffea follows [CalVer](https://calver.org/). Maintainers cut releases as
needed and publish them on the GitHub Releases page. Changes that land in
`master` will be part of the next scheduled release.

Thank you for helping improve coffea! :coffee:
