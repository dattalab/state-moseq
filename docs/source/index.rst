State MoSeq
===========


.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - TODO: Paper
     - `GitHub <https://github.com/dattalab/state-moseq/>`_
     - `License <https://github.com/dattalab/state-moseq/blob/main/LICENSE.md>`_


State MoSeq (sMoSeq) is a method for discovering high-level states in animal behavior. Given low-level behavior labels (e.g. MoSeq syllables), sMoSeq fits a hierarchical hidden Markov model to identify how behaviors are clustered over time. 


Installation
------------

.. note::

   If using Windows, make sure to run all the commands below from an Anaconda Prompt.

.. note::

   State moseq supports the same platforms as `jax <https://github.com/jax-ml/jax?tab=readme-ov-file#supported-platforms>`_. That is, it supports CPU and GPU installations on linux systems, and CPU installations on MacOS and Windows systems. GPU on WSL2 is considered 'experimental'.

Create a new conda environment with python 3.10::

   conda create -n state_moseq python=3.10
   conda activate state_moseq

Then use pip to install the version of state moseq that you want::

   pip install state-moseq # CPU only
   pip install state-moseq[cuda] # NVIDIA GPU

To run state-moseq in jupyter, either launch jupyterlab directly from the ``state_moseq`` environment or register a globally-accessible jupyter kernel as follows::

   python -m ipykernel install --user --name=state_moseq


.. toctree::
   :caption: Practical tutorials
   :maxdepth: 1

   example_data_tutorial
   parameter_scan_tutorial

.. toctree::
   :caption: Pedagogical tutorials
   :maxdepth: 1

   standard_hhmm_tutorial
   efficient_hhmm_tutorial

.. toctree::
   :caption: Developer API
   :maxdepth: 1

   hhmm_efficient
   hhmm_standard
   utils
   viz