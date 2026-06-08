Installation
============

Requirements
------------

- Python 3.10 or newer
- An OpenAI API key for transcription
- Optional: a Hugging Face token and ``pyannote.audio`` for speaker diarization

Install with uv
---------------

.. code-block:: bash

   uv sync

Install with pip
----------------

.. code-block:: bash

   pip install -e .

Optional diarization dependencies
---------------------------------

.. code-block:: bash

   uv sync --extra diarize

or:

.. code-block:: bash

   pip install -e ".[diarize]"

Configuration
-------------

Set credentials in the environment:

.. code-block:: bash

   export OPENAI_API_KEY="your_api_key_here"
   export HUGGINGFACE_TOKEN="your_huggingface_token_here"

``whisper-smith`` also loads a ``.env`` file from the project root.
