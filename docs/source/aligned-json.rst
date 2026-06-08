Aligned JSON Workflow
=====================

The aligned JSON workflow is the main product flow for combining transcription
and diarization.

Run the pipeline
----------------

.. code-block:: bash

   whisper-smith data/sample.m4a --align --output data/sample.aligned.json

Outputs:

.. code-block:: text

   data/sample.aligned.json
   data/sample.transcript.json
   data/sample.diarization.json

Output shape
------------

Each aligned transcript segment contains timestamps, text, and the assigned
speaker label:

.. code-block:: json

   {
     "segments": [
       {
         "start": 0.0,
         "end": 7.08,
         "text": "Hello world.",
         "speaker": "SPEAKER_01"
       }
     ],
     "text": "Hello world."
   }

How speakers are assigned
-------------------------

``assign_speakers`` compares each transcript segment with diarization segments
and chooses the speaker with the largest time overlap. If no diarization segment
overlaps, the transcript segment keeps its existing speaker value.
