Google Colab: Aligned Transcript
=================================

Running ``whisper-smith`` on Google Colab gives you a free GPU to speed up
speaker diarization and lets you process audio without installing anything locally.

.. note::

   Before running, go to **Runtime → Change runtime type** and select **T4 GPU**.

Prerequisites
-------------

In the Colab sidebar open **Secrets** (key icon) and add:

- ``OPENAI_API_KEY`` — your OpenAI API key
- ``HUGGINGFACE_TOKEN`` — your Hugging Face token

You must also accept the `pyannote/speaker-diarization-3.1
<https://huggingface.co/pyannote/speaker-diarization-3.1>`_ model terms
on Hugging Face before the pipeline can be downloaded.

Open in Colab
-------------

Click the badge to open the notebook directly in Google Colab — no download needed:

.. raw:: html

   <a href="https://colab.research.google.com/github/yeiichi/whisper-smith/blob/main/notebooks/colab_aligned_transcript.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
   </a>

The notebook is also available in the repository at:

.. code-block:: text

   notebooks/colab_aligned_transcript.ipynb

Notebook walkthrough
--------------------

**Step 1 — Install whisper-smith**

.. code-block:: python

   %pip install -q "whisper-smith[diarize]"

**Step 2 — Load credentials from Colab Secrets**

.. code-block:: python

   import os
   from google.colab import userdata

   os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
   os.environ["HUGGINGFACE_TOKEN"] = userdata.get("HUGGINGFACE_TOKEN")

**Step 3 — Upload audio**

.. code-block:: python

   from google.colab import files
   from pathlib import Path

   uploaded = files.upload()
   audio_path = Path(next(iter(uploaded)))
   output_path = audio_path.with_suffix(".aligned.json")

**Step 4 — Run the pipeline**

This is the direct Colab equivalent of the local CLI command:

.. code-block:: bash

   whisper-smith audio.m4a --align --output audio.aligned.json

In the notebook, the uploaded filename is substituted at runtime:

.. code-block:: python

   !whisper-smith "{audio_path}" --align --output "{output_path}"

**Step 5 — Preview results**

.. code-block:: python

   import json

   data = json.loads(output_path.read_text(encoding="utf-8"))
   for seg in data["segments"]:
       speaker = seg.get("speaker") or "UNKNOWN"
       print(f"[{seg['start']:6.2f}s – {seg['end']:6.2f}s]  {speaker:12s}  {seg['text'].strip()}")

**Step 6 — Download the JSON**

.. code-block:: python

   files.download(str(output_path))

Advanced: explicit GPU pipeline
--------------------------------

For fine-grained control — such as specifying the number of speakers — load
the pyannote pipeline manually, move it to the GPU with ``.to(device)``, and
pass it via the ``pipeline=`` argument. This also avoids re-downloading the
model if you run diarization multiple times in the same session.

.. code-block:: python

   import torch
   from pyannote.audio import Pipeline
   from whisper_smith.align import assign_speakers
   from whisper_smith.diarize import diarize_audio
   from whisper_smith.exporters import export_json
   from whisper_smith.transcribe import transcribe_audio

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   diarize_pipeline = Pipeline.from_pretrained(
       "pyannote/speaker-diarization-3.1",
       token=os.environ["HUGGINGFACE_TOKEN"],
   )
   diarize_pipeline.to(device)

   transcript  = transcribe_audio(audio_path)
   diarization = diarize_audio(audio_path, pipeline=diarize_pipeline, num_speakers=2)
   aligned     = assign_speakers(transcript, diarization)

   output_path.write_text(export_json(aligned), encoding="utf-8")
