# Installation

This page is focused on detailing how to access ``coffea`` in a variety of ways.
The following quick start is a good place to start for local development, but
it is _not_ a helpful method for long-term analysis and scale out on clusters.

## Quick start

Coffea is a python package distributed via [PyPI](https://pypi.org/project/coffea/).
A python installation version 3.10 or newer is required to use coffea.
```
pip install coffea
```

## Platform Support

coffea core functionality is routinely tested on Windows, Linux and macOS.
All [local executors](./concepts.md#local-executors) are tested against all three platforms,
however the [distributed executors](./concepts.md#distributed-executors) are not routinely tested on Windows.

Coffea starts from v0.5.0 in the PyPI repository since before v0.5.0 it was hosted as [fnal-column-analysis-tools](https://pypi.org/project/fnal-column-analysis-tools/). If you are still using fnal-column-analysis-tools, please move to [coffea](https://pypi.org/project/coffea/)!
In April 2023, Coffea moved to calendar versioning with the last semantic version being v0.7.x.
If you are still using the last semantic version, please move to a more recent calendar version since
we will stop backporting features and patches to v0.7 soon.

## Install with pip (or equivalent)

The quick start's short command hides many different mostly-equivalent options:

- install coffea system-wide using `pip install coffea`;
- if you do not have administrator permissions, install as local user with `pip install --user coffea`;
- for longer-term stability and reproducibility, you can set up a [virtual environment](#virtual-environment)
- if you use [Conda](https://docs.conda.io/projects/conda/en/latest/index.html), simply `conda install coffea`;

To update a previously installed coffea to a newer version, use: `pip install --upgrade coffea`
Although not required, it is recommended to also [install Jupyter](https://jupyter.org/install), as it provides a more interactive development environment.
The installation procedure is essentially identical as above: `pip install jupyter`. (If you use conda, `conda install jupyter` is a better option.)

:::{note}
In rare cases, you may find that the `pip` executable in your path does not correspond to the same python installation as the `python` executable. This is a sign of a broken python environment. However, this can be bypassed by using the syntax `python -m pip ...` in place of `pip ...`.
:::

### Optional dependencies

Coffea supports several optional components that require additional package installations.
In particular, all of the [distributed executors](./concepts.md#distributed-executors) require additional packages.
The necessary dependencies can be installed easily via ``pip`` using the setuptools [optional dependencies](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies) facility:

   - [parsl](https://parsl-project.org/) distributed executor: ``pip install coffea[parsl]``
   - [dask](https://distributed.dask.org/en/latest/) distributed executor: ``pip install coffea[dask]``
   - [dask-awkward](https://dask-awkward.readthedocs.io/en/stable/) and [dask-histogram](https://dask-histogram.readthedocs.io/en/stable/) support: ``pip install coffea[dask-awkward]``
   - [TaskVine](https://ccl.cse.nd.edu/software/taskvine/) distributed executor: see the installation guide in their docs and use the `TaskVineExecutor` in coffea.

Multiple extras can be installed together via, e.g. `pip install coffea[dask,dask-awkward,parsl]`

(virtual-environment)=
### Virtual environment

Virtual environments are a good way to isolate python environments (so that two different projects can have potentially-conflicting dependencies) and ensure no hidden dependencies.
You can find more information at [`venv`](https://docs.python.org/3/library/venv.html).

```bash
python -m venv my_env
source my_env/bin/activate
pip install coffea
```

(pre-built-images)=
## Use Coffea with Pre-Built Images

Official images are maintained at the [CoffeaTeam/af-images](https://github.com/CoffeaTeam/af-images) repository and available on DockerHub.
In order to be able to run these container images, you will need a container "runner".
On your personal computer, you can install [docker](https://docs.docker.com/engine/install/) or [podman](https://podman.io/).
On computing clusters, you should check for the [``apptainer``](https://apptainer.org/) command and ask the cluster administrators to install it if it does not exist.

A container image often has an associated "tag" that helps us humans understand the purpose of the image.
You should select a tag that works for your purposes from [the list of options below](#image-naming).

(image-naming)=
### Image Names
In general, container image names look like ``user/repo:tag``.
The repositories are hosted under the ``coffeateam`` user on DockerHub, so all of the official
Coffea images have ``coffeateam`` as the "user".
The repository name depends on which features and what base operating system is in use.

repo | description
-----|-------------
``coffea-dask-almalinux8`` | general purpose Coffea in AlmaLinux 8
``coffea-dask-almalinux9`` | general purpose Coffea in AlmaLinux 9
``coffea-dask-almalinux9-noml`` | No Machine Learning libraries (smaller image, easier to copy)
``coffea-dask-almalinux9-eaf`` | Including Execute Ahead Framework (EAF)
``coffea-base-almalinux8`` | legacy image for 0.7.x Coffea in AlmaLinux 8
``coffea-base-almalinux9`` | legacy image for 0.7.x Coffea in AlmaLinux 9

The tags within each of these repositories have the following form.

- `latest`: Current stable release (recommended for most users)
- `latest-py3.X`: Latest stable release with specific Python version (3.8, 3.9, 3.10, 3.11, 3.12)
- `202X.X.X-py3.XX`: Specific calendar-versioned release with Python version
- `dev`: Development branch (unstable, only use if actively testing developments)
- `head`: Main branch (unstable, only use if actively testing developments)

As an example, the recommended full image name (and the one in the [how to run](#running-image) information below) is
```
coffeateam/coffea-dask-almalinux9:latest
```

For a complete list of all available images, visit [DockerHub](https://hub.docker.com/u/coffeateam).

(running-image)=
### How to Run
Containers are a general purpose technology and they have many features.
We are just using them to help isolate the Coffea running environment from the system installation of Python and other Python packages.

Remember, I am just using ``coffeateam/coffea-dask-almalinux9:latest`` as an example.
It is a good default to use, but you should consider using a different image if you want to
pin to a specific Coffea/Python/AlmaLinux version.

#### Docker (or Podman)
Both Docker and Podman have a similar interface and so they can be run in a similar way.
```bash
docker run -it --rm --name coffea-container coffeateam/coffea-dask-almalinux9:latest
# replace `docker` with `podman` if you installed podman
```

:::{note}
This command should be run from _within_ WSL if you are using Windoze.
:::

### Apptainer (formerly Singularity)
For the following, I will use the newer name ``apptainer`` but these features will function with the old name ``singularity``.

:::{tip}
You will need extra mounts from the host into the container environment
in order to use a "Grid Certificate" to maintain access to large samples.

The additional mounts are
- `${HOME}`: where your certificates (probably under `~/.globus`) and proxy files (probably look like `~/x509<stuff>`) live
- `/etc/grid-security`: extra configuration for the `voms-*` tools so they know how to connect to the authentication network

You can add more mounts to `apptainer` after the `-B` flag (short for "bind mount")
with a comma separate list.

You need to `voms-proxy-init` _outside_ of the container and then these mounts let
you share this initialized-proxy with the environment _inside_ the container.
For example
```
$ voms-proxy-init --voms cms --rfc --valid 192:0
$ voms-proxy-info
# details of current proxy
$ apptainer shell -B ${PWD},${HOME},/cvmfs/cms.cern.ch/etc/grid-security:/etc/grid-security \
    /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:latest
$ voms-proxy-info
# returns same result as earlier except timeleft field
```

:::

If your cluster has ``apptainer`` and the ``/cvmfs/unpacked.cern.ch`` directory mounted,
then you can run the images that are already distributed via CVMFS (saving you time and disk space).
```bash
apptainer shell -B ${PWD}:/work \
    /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:latest
```
Only the stuff after ``coffeateam`` needs to change if you are using a different image.

:::{warning}
The ``latest`` image tag on CVMFS automatically updates when there is a new release, but
this is different from the behavior when using Docker/Podman (or downloading the image
yourself below) where the image is only downloaded if it doesn't already exist (or if you
manually call the ``pull`` command).
:::

If your cluster does not have CVMFS enables or the `unpacked.cern.ch` CVMFS repository mounted,
you can still run the image, you will just need to download a copy of it yourself.
``apptainer`` does this automatically for you, but it puts the downloaded image into your home
directory if you do not define ``APPTAINER_CACHEDIR``. Since most clusters have a restricted
size allocated to your home directory, it is recommended to define ``APPTAINER_CACHEDIR`` to
be a larger and more permanent location compared to your home directory.
This definition should go into your shell initialization file (e.g. ``~/.bashrc``) and then
you can run ``apptainer`` like
```bash
apptainer shell -B ${PWD}:/work docker://coffeateam/coffea-dask-almalinux9:latest
```
As before, only the stuff after ``coffeateam`` needs to change if you are using a different image.

## For Developers

1. Download source:

   ```bash
   git clone https://github.com/scikit-hep/coffea
   ```

2. Install with development dependencies. These live in a
   [dependency group](https://packaging.python.org/en/latest/specifications/dependency-groups/),
   so `pip install --group` requires pip 25.1 or newer:

   ```bash
   cd coffea
   pip install --editable . --group dev
   # or if you need to work on the executors, e.g. dask,
   pip install --editable '.[dask]' --group dev
   ```

   If you use [`uv`](https://docs.astral.sh/uv/) to manage your python environments, you can also simply run
   `uv sync` to install the development dependencies.

3. Develop a cool new feature or fix some bugs

4. Lint source, run tests, and build documentation:

   ```bash
   pre-commit run --all-files
   pytest tests
   pushd docs && make html && popd
   ```

5. Make a pull request!
