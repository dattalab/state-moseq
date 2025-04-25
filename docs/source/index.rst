State MoSeq
===========


.. list-table::
   :widths: 30 30 30 30
   :header-rows: 0

   * - `GitHub <https://github.com/dattalab/state-moseq/>`_
     - TODO: Colab
     - TODO: Paper
     - `License <https://github.com/dattalab/state-moseq/blob/main/LICENSE.md>`_

State MoSeq (sMoSeq) is a method for discovering higher-order states in animal behavior. Given low-level behavior labels (e.g. MoSeq syllables), sMoSeq fits a hierarchical hidden Markov model to identify how behaviors are clustered over time. 


.. toctree::
   :caption: Setup
   
   install
   TODO colab <https://colab.research.google.com/github/dattalab/keypoint-moseq/blob/main/docs/keypoint_moseq_colab.ipynb>


.. toctree::
   :caption: Tutorials

   example_data_tutorial
   standard_hhmm_tutorial
   efficient_hhmm_tutorial

.. toctree::
   :caption: Developer API

   hhmm_efficient
   hhmm_standard
   utils