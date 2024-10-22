## How to contribute to coffea

#### **Did you find a bug?**

* **Ensure the bug was not already reported** by searching on GitHub under [Issues](https://github.com/CoffeaTeam/coffea/issues).

* If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/CoffeaTeam/coffea/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

#### **Do you want to write a patch that fixes a bug?**

* Follow the [setup instructions for developers](https://coffeateam.github.io/coffea/installation.html#for-developers) to get a development environment

* Open a new GitHub pull request with the patch.

* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

* Before submitting, please run `pre-commit run --all-files` and `pytest` to ensure you follow our formatting conventions and do not break any existing code. Furthermore, we prefer that any newly contributed code does not reduce the current code coverage of the repository. Please make sure your test your code as thoroughly as is needed.

#### **Did you fix whitespace, format code, or make a purely cosmetic patch?**

Changes that are cosmetic in nature and do not add anything substantial to the stability, functionality, or testability of coffea will generally not be accepted.

#### **Do you intend to add a new feature or change an existing one?**

* Suggest your change either in a new GitHub feature request [issue](https://github.com/CoffeaTeam/coffea/issues) or in the [Discussions](https://github.com/CoffeaTeam/coffea/discussions) section!

#### **Do you have questions about the source code?**

* Ask any question about how to use coffea in the [coffea iris-hep slack channel](https://iris-hep.slack.com) or in the [Discussions](https://github.com/CoffeaTeam/coffea/discussions) section.

#### **Do you want to contribute to the coffea documentation?**

* Follow the [setup instructions for developers](https://coffeateam.github.io/coffea/installation.html#for-developers) to get a development environment

* Edit the ReStructured Text files in `docs/source` or the docstrings in the python source code as appropriate

* Run `pushd docs && make html && popd` to compile, and open `docs/build/html/index.html` in a browser to see your local changes

coffea is a HEP community and volunteer effort. We encourage you to pitch in and [join the team](mailto:cms-coffea@cern.ch)!

* Fixes, changes, and documentation updates will be released in a timely manner. Coffea follows [CalVer](https://calver.org/) practices. Repository maintainers will generate new releases as necessary, and releases are made using the github release pages.

Thanks! :coffee: :coffee: :coffee:

Coffea Team
