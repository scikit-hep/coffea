# How to distribute additional packages during scale out

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
For example, while you may like using `zfit` to fit different models to the resulting histograms,
the fitting package (and all its heavier dependencies) is not needed while the histograms are being filled.
You can leave that out of your batch environment to keep it smaller.

In most cases, you will not even need to include additional packages outside of the pre-built image
for the batch jobs. In these cases, you do not need to copy an environment.
:::

For example, the following code starts from the `coffea-dask-almalinux8` image
and adds a special python module that is not included in the base image:

```bash
singularity shell -B ${PWD}:/srv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux8:latest
cd /srv
python -m venv --without-pip --system-site-packages myenv
source myenv/bin/activate
python -m pip install zfit
```

This creates a virtual environment `myenv` and a directory with the same name where the extra python module will be
installed. At this point, the terminal prompt will look like `(myenv) Singularity>`, indicating you are inside a Singularity
image and have `myenv` activated.
Next time you log in, only lines 1, 2, and 4 need to be re-executed.

If using HTCondor for job submission, you can create a tarball of the virtual environment directory and then submit condor
jobs using the `+SingularityImage` [HTCondor option](https://htcondor.readthedocs.io/en/latest/admin-manual/ep-policy-configuration.html#container-vm-support-docker-apptainer-singularity-and-xen-vmware).
Note that this option is not enabled by default in HTCondor installations, so you may need to talk to your site administrator to be
able to use this option (if they do not want to enable this option, you can also run `apptainer` within your wrapper script as well).
You will also need to create a small wrapper script to unpack and re-source the environment to have the job use the
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
