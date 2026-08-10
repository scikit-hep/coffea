.. _intro-coffea-wq:

Work Queue Executor
===================

`Work Queue <https://cctools.readthedocs.io/en/latest/work_queue/>`_ is a
distributed computing framework used to build large scale manager-worker
applications, developed by the Cooperative Computing Lab
(CCL) at the University of Notre Dame. This executor functions as the
manager program which divides up a Coffea data analysis workload into
discrete tasks.  A large number of worker processes running on
cluster or cloud systems will execute the tasks.

To set up Coffea and Work Queue together, you may need to
create a Conda environment, install the software, and then
create a tarball containing the environment.  The tarball is
sent to each worker in order to provide the same environment
as the manager machine.

.. code-block:: bash

  # Create a new environment
  conda create --yes --name coffea-env -c conda-forge python coffea xrootd ndcctools conda conda-pack
  conda activate coffea-env

  # Pack the environment into a portable tarball.
  conda-pack --output coffea-env.tar.gz

To run an analysis, you must set up a work queue executor
with appropriate arguments.  Here is a complete example:

.. literalinclude:: wq-example.py
   :language: Python

When executing this example,
you should see that Coffea begins to run, and a progress bar
shows the creation of tasks.  Workers are created locally using the factory
declared.

You can also launch workers outside python. For testing purposes, you can start
a single worker on the same machine, and direct it to connect to your manager
process, like this:

.. code-block::

  work_queue_worker -P password.txt <hostname> 9123

Or:

.. code-block::

  work_queue_worker -P password.txt -M coffea-wq-${USER}

With a single worker, the process will be gradual as it completes
one task (or a few tasks) at a time.  The output will be similar to this:

.. code-block::

  ------------------------------------------------
  Example Coffea Analysis with Work Queue Executor
  ------------------------------------------------
  Manager Name: -M coffea-wq-btovar
  ------------------------------------------------
  Listening for work queue workers on port 9123.
  submitted preprocessing task id 1 item pre_0, with 1 file
  submitted preprocessing task id 2 item pre_1, with 1 file
  preprocessing task id 2 item pre_1 with 1 events on localhost. return code 0 (success)
  allocated cores: 2.0, memory: 1000 MB, disk 2000 MB, gpus: 0.0
  measured cores: 0.3, memory: 120 MB, disk 6 MB, gpus: 0.0, runtime 3.1 s
  preprocessing task id 1 item pre_0 with 1 events on localhost. return code 0 (success)
  allocated cores: 2.0, memory: 1000 MB, disk 2000 MB, gpus: 0.0
  measured cores: 0.3, memory: 120 MB, disk 6 MB, gpus: 0.0, runtime 2.9 s
  submitted processing task id 3 item p_2, with 100056 event
  submitted processing task id 4 item p_3, with 100056 event
  submitted processing task id 5 item p_4, with 100056 event
  submitted processing task id 6 item p_5, with 100056 event
  Preprocessing 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━      2/2 [ 0:00:06 < 0:00:00 | 0.3  file/s ]
      Submitted   0%                                  0/400224 [ 0:00:00 < -:--:-- | ?   event/s ]
      Processed   0%                                  0/400224 [ 0:00:00 < -:--:-- | ?   event/s ]
      Accumulated 0%                                       0/1 [ 0:00:00 < -:--:-- | ?   tasks/s ]


To run at larger scale, you will need to run a large number
of workers on a cluster or other infrastructure.  For example,
to submit 32 workers to an HTCondor pool:

.. code-block::

  condor_submit_workers -M coffea-wq-${USER} -P password.txt 1


Similarly, you can run the worker's factory outside the manager. In that way,
you can have the manager and the factory running on different machines:

.. code-block::

  work_queue_factory -T condor -M coffea-wq-${USER} -P password.txt --max-workers 10 --cores 8 --python-env=env.tar.gz

For more information on starting and managing workers
on various batch systems and clusters, see the
`Work Queue <https://cctools.readthedocs.io/en/latest/work_queue/>`_ documentation
