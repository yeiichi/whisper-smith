CLI Guide
=========

Basic usage
-----------

.. code-block:: bash

   whisper-smith data/sample.m4a

Save transcript output:

.. code-block:: bash

   whisper-smith data/sample.m4a --output data/sample.txt

Choose an output format:

.. code-block:: bash

   whisper-smith data/sample.m4a --format json --output data/sample.json

Supported transcript formats are ``txt``, ``json``, ``srt``, and ``vtt``.
When ``--format`` is omitted, the format is inferred from ``--output``.

Overwrite an existing output file:

.. code-block:: bash

   whisper-smith data/sample.m4a --output data/sample.txt --overwrite

Speaker diarization
-------------------

.. code-block:: bash

   whisper-smith data/sample.m4a --diarize --output data/sample.diarization.json

Diarization output currently supports JSON only.

Optional speaker-count hints:

.. code-block:: bash

   whisper-smith data/sample.m4a --diarize --num-speakers 2
   whisper-smith data/sample.m4a --diarize --min-speakers 1 --max-speakers 3

Speaker-aligned JSON
--------------------

.. code-block:: bash

   whisper-smith data/sample.m4a --align --output data/sample.aligned.json

This writes the aligned transcript JSON as the main output and writes
intermediate transcript and diarization JSON files beside it.

Use a separate artifact directory:

.. code-block:: bash

   whisper-smith data/sample.m4a --align --output data/sample.aligned.json --artifacts-dir data/artifacts

Suppress intermediate files:

.. code-block:: bash

   whisper-smith data/sample.m4a --align --output data/sample.aligned.json --no-artifacts
