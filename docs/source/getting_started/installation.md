# Installation

This page is focused on detailing how to access ``coffea`` in a variety of ways.
The following quick start is a good place to start for local development, but
it is _not_ a helpful method for long-term analysis and scale out on clusters.

## Quick start

Coffea is a python package distributed via [PyPI](https://pypi.org/project/coffea/).
A python installation version 3.6 or newer is required to use coffea.
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
Coffea images have ``coffea`` team as the "user".
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
- `202X.X.X-pyX.XX`: Specific calendar-versioned release with Python version
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

:::{tip}
The following commands are long and arduous to type out.
You may find [``denv``](https://tomeichlersmith.github.io/denv/) to be a helpful program to install on both your
personal computer and on the cluster(s) you work on in order to handle
the switch between docker/podman/apptainer for you.

On both your personal computer and a remote cluster, you would choose an image
```
denv init coffeateam/coffea-dask-almalinux9:latest
```
and then run from within this image
```
denv python my-analysis.py
```
with the image choice being stored in a local configuration file.
:::

TODO: comment about ``vomx-proxy-init`` and how that needs to be done from within the container environment
either when using ``denv`` or when using ``apptainer`` directly.

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

If your cluster does not have CVFMS enables or the `unpacked.cern.ch` CVMFS repository mounted,
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

## Creating a portable virtual environment

In some instances, it may be useful to have a self-contained environment that can be relocated.
One use case is for users of coffea that do not have access to a distributed compute cluster that is compatible with
one of the coffea [distributed executors](./concepts.md#distributed-executors). Here, a fallback solution can be found by creating traditional batch jobs (e.g. HTCondor)
which then use coffea [local executors](./concepts.md#local-executors), possibly multi-threaded. In this case, often the user-local Python package directory
is not available from batch workers, so a portable python environment needs to be created.
Annoyingly, Python virtual environments are not portable by default due to several hardcoded paths in specific locations, however
there are two workarounds presented below. In both cases, we make a virtual environment that starts from a non-system base
python environment to lower the amount of needed installations in the virtual environment. One can always start a venv from scratch,
but the number of coffea dependencies makes the installation rather large, up to a few hundred MB.


### Container-based

If we start from one of the images in `/cvmfs/unpacked.cern.ch/` from the [pre-built images](#pre-built-images) section, we don't have to install nearly as much
software in our virtual environment, letting the container image take care of the majority of the codebase.

:::{tip}
Consider if your extra packages need to be included in the environment that is run within the batch jobs.
For example, while `matplotlib` is helpful for creating plots from a set of histograms, you could leave
that out of your batch environment to keep it smaller.

In many cases, you will not even need to include additional packages outside of the pre-built image
for the batch jobs. In these cases, you do not need to copy an environment.
:::

For example, the following code starts from the `coffea-dask-almalinux8` image
and adds a special python module that is not included in the base image:

```bash
singularity shell -B ${PWD}:/srv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux8:latest
cd /srv
python -m venv --without-pip --system-site-packages myenv
source myenv/bin/activate
python -m pip install --ignore-installed h5py
```

This creates a virtual environment `myenv` and a directory with the same name where the extra python module `h5py` will be
installed. At this point, the terminal prompt will look like `(myenv) Singularity>`, indicating you are inside a Singularity
image and have `myenv` activated.
Next time you log in, only lines 1, 2, and 4 need to be re-executed.

If using HTCondor for job submission, you can create a tarball of the virtual environment directory and then submit condor
jobs using the `+SingularityImage` [HTCondor option](https://htcondor.readthedocs.io/en/latest/admin-manual/ep-policy-configuration.html#container-vm-support-docker-apptainer-singularity-and-xen-vmware).
Note that this option is not enabled by default in HTCondor installations, so you may need to talk to your site administrator to be
able to use this option. You will also need to create a small wrapper script to re-source the environment to have the job use the
same environment as your interactive container.
A complete example that runs at FNAL LPC is shown [in this gist](https://gist.github.com/mattbellis/20b9f892689c8a32b99151c5aa7a4e5f).

### LCG-Based

Although the local installation can work anywhere, if the base environment does not already have most of the coffea dependencies, then the user-local package directory can become quite bloated.
An option to avoid this bloat is to use a base Python environment provided via [CERN LCG](https://lcginfo.cern.ch/), which is available on any system that has the [CVMFS](https://cvmfs.readthedocs.io/en/stable/) directory `/cvmfs/sft.cern.ch/` mounted.
Simply source a LCG release (shown here: 98python3) and install:

```bash
# check your platform: CC7 shown below, for SL6 it would be "x86_64-slc6-gcc8-opt"
source /cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc9-opt/setup.sh  # or .csh, etc.
pip install --user coffea
```

:::{danger}
This method can be fragile, since the LCG-distributed packages may conflict with the coffea dependencies. In general it is better to define your own environment or use an image.
:::

There are not many locations to edit to make a venv portable, and some sed hacks can save the day.
Here is an example of a bash script that installs coffea on top of the LCG 98python3 software stack inside a portable virtual environment,
with the caveat that cvmfs must be visible from batch workers:

```bash
#!/usr/bin/env bash
NAME=coffeaenv
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc9-opt

source $LCG/setup.sh
# following https://aarongorka.com/blog/portable-virtualenv/, an alternative is https://github.com/pex-tool/pex
python -m venv --copies $NAME
source $NAME/bin/activate
LOCALPATH=$NAME$(python -c 'import sys; print(f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")')
export PYTHONPATH=${LOCALPATH}:$PYTHONPATH
python -m pip install setuptools pip wheel --upgrade
python -m pip install coffea
sed -i '1s/#!.*python$/#!\/usr\/bin\/env python/' $NAME/bin/*
sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $NAME/bin/activate
sed -i "2a source ${LCG}/setup.sh" $NAME/bin/activate
sed -i "3a export PYTHONPATH=${LOCALPATH}:\$PYTHONPATH" $NAME/bin/activate
tar -zcf ${NAME}.tar.gz ${NAME}
```

The resulting tarball size is about 60 MB.
An example batch job wrapper script is:

```bash
#!/usr/bin/env bash
tar -zxf coffeaenv.tar.gz
source coffeaenv/bin/activate

echo "Running command:" $@
time $@ || exit $?
```

Note that this environment only functions from the working directory of the wrapper script due to having relative paths.
Unless you install jupyter into this environment (which may bloat the tarball--LCG98 jupyter is reasonably recent), it is not visible inside the LCG jupyter server. From a shell with the virtual environment activated, you can execute:

```bash
python -m ipykernel install --user --name=coffeaenv
```

to make a new kernel available that uses this environment.

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
