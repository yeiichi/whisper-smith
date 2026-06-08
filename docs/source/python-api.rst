Python Usage
============

Transcribe audio
----------------

.. code-block:: python

   from pathlib import Path

   from whisper_smith.exporters import export_transcript
   from whisper_smith.transcribe import transcribe_audio

   result = transcribe_audio(Path("data/sample.m4a"))
   print(result.text)

   srt = export_transcript(result, "srt")
   Path("data/sample.srt").write_text(srt, encoding="utf-8")

Diarize audio
-------------

.. code-block:: python

   from pathlib import Path

   from whisper_smith.diarize import diarize_audio

   result = diarize_audio(Path("data/sample.m4a"))

   for segment in result.segments:
       print(segment.start, segment.end, segment.speaker)

Assign speakers
---------------

.. code-block:: python

   from whisper_smith.align import assign_speakers

   aligned = assign_speakers(transcript, diarization)

Export JSON
-----------

.. code-block:: python

   from pathlib import Path

   from whisper_smith.exporters import export_json

   Path("data/sample.aligned.json").write_text(
       export_json(aligned),
       encoding="utf-8",
   )
