Troubleshooting
===============

TorchAudio deprecation warnings
-------------------------------

When running diarization, TorchAudio may print deprecation warnings about
``list_audio_backends`` or TorchCodec. These warnings are expected with the
compatible diarization dependency stack and are not fatal.

``torchaudio`` missing ``AudioMetaData``
----------------------------------------

If diarization fails because ``torchaudio`` has no ``AudioMetaData`` attribute,
refresh the optional diarization dependencies:

.. code-block:: bash

   uv lock --upgrade-package torch --upgrade-package torchaudio
   uv sync --extra diarize

PyTorch checkpoint safe globals
-------------------------------

PyTorch 2.6 and newer default to safer ``weights_only`` checkpoint loading.
``whisper-smith`` registers the trusted pyannote checkpoint metadata classes it
needs before loading the diarization pipeline.

Hugging Face model access
-------------------------

If the diarization model cannot load, check that ``HUGGINGFACE_TOKEN`` is set
and that the model's Hugging Face user conditions have been accepted.
