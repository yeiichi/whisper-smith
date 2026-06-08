whisper-smith
=============

``whisper-smith`` is a small Python CLI and library for transcribing audio,
running speaker diarization, and producing speaker-aligned transcript JSON.

The main workflow is:

.. code-block:: text

   sound file -> transcript JSON + diarization JSON -> aligned JSON

No local setup? Run the full pipeline on a free GPU in Google Colab:

.. raw:: html

   <a href="https://colab.research.google.com/github/yeiichi/whisper-smith/blob/main/notebooks/colab_aligned_transcript.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
   </a>

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   cli
   aligned-json
   python-api
   colab
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
