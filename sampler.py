import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from typing import Union

from voicecloner.encoder import inference as encoder
from voicecloner.encoder.params_model import model_embedding_size as speaker_embedding_size
from voicecloner.synthesizer.inference import Synthesizer
from voicecloner.utils.default_models import ensure_default_models
from voicecloner.vocoder import inference as vocoder

class VoiceCloner:

    def __init__(self,
                save_model_root="saved_models",
                enc_model_fpath="saved_models/default/encoder.pt",
                syn_model_fpath="saved_models/default/synthesizer.pt",
                voc_model_fpath="saved_models/default/vocoder.pt",
                cpu=False,
                seed=None,
                verbose=False):
        """
        Load the models one by one.
        """

        # Hide GPUs from Pytorch to force CPU processing
        if cpu: os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        enc_model_fpath = Path(enc_model_fpath)
        syn_model_fpath = Path(syn_model_fpath)
        voc_model_fpath = Path(voc_model_fpath)
        save_model_root = Path(save_model_root)

        if verbose: print("Loading models...\n")

        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device_id)
            ## Print some environment information (for debugging purposes)
            if verbose: print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
                "%.1fGb total memory.\n" %
                (torch.cuda.device_count(),
                device_id,
                gpu_properties.name,
                gpu_properties.major,
                gpu_properties.minor,
                gpu_properties.total_memory / 1e9))
        else:
            if verbose: print("Using CPU for inference.\n")

        ## Load the models one by one.
        if verbose: print("Preparing the encoder, the synthesizer and the vocoder...")
        ensure_default_models(save_model_root)
        encoder.load_model(enc_model_fpath)
        synthesizer = Synthesizer(syn_model_fpath)
        vocoder.load_model(voc_model_fpath)

        self.device_id = device_id
        self.synthesizer = synthesizer
        self.seed = seed
        self.voc_model_fpath = voc_model_fpath
        self.syn_model_fpath = syn_model_fpath
        self.style_embed = None
        self.verbose = verbose

        self._test_models()

    def embed_style(self,
                    in_fpath: str):
        """
        Given a path to an audio file, generates the speaker embedding.
        """

        # load audio file
        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        original_wav, sampling_rate = librosa.load(str(in_fpath))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        if self.verbose: print("Loaded file succesfully")

        # Then we derive the embedding. There are many functions and parameters that the
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        emb = encoder.embed_utterance(preprocessed_wav)
        if self.verbose: print("Created the embedding")

        return emb

    def voice_clone(self,
                    text: str,
                    emb: np.ndarray,
                    out_fpath: Union[str, None] = None):
        """
        Given the previously generated speaker embedding, synthesizes text in the voice of
        """

        # reset synthesizer
        if self.seed is not None:
            torch.manual_seed(self.seed)
            self.synthesizer = Synthesizer(self.syn_model_fpath)
            
        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        embeds = [emb]

        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        
        if self.verbose: print("Created the mel spectrogram")

        ## Generating the waveform
        if self.verbose: print("Synthesizing the waveform:")

        # If seed is specified, reset torch seed and reload vocoder
        if self.seed is not None:
            torch.manual_seed(self.seed)
            vocoder.load_model(self.voc_model_fpath)

        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        generated_wav = vocoder.infer_waveform(spec)

        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")

        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        generated_wav = encoder.preprocess_wav(generated_wav)

        if self.verbose: print("Sample complete!")

        # Save it on the disk
        if out_fpath is not None:
            sf.write(out_fpath, generated_wav.astype(np.float32), self.synthesizer.sample_rate)
            if self.verbose: print("\nSaved output as %s\n\n" % out_fpath)

        return generated_wav
    
    def _test_models(self):

        ## Run a test of loaded models
        if self.verbose: print("Testing your configuration with small inputs.")
        # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
        # sampling rate, which may differ.
        # If you're unfamiliar with digital audio, know that it is encoded as an array of floats
        # (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
        # The sampling rate is the number of values (samples) recorded per second, it is set to
        # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
        # to an audio of 1 second.
        if self.verbose: print("\tTesting the encoder...")
        encoder.embed_utterance(np.zeros(encoder.sampling_rate))

        # Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
        # returns, but here we're going to make one ourselves just for the sake of showing that it's
        # possible.
        embed = np.random.rand(speaker_embedding_size)
        # Embeddings are L2-normalized (this isn't important here, but if you want to make your own
        # embeddings it will be).
        embed /= np.linalg.norm(embed)
        # The synthesizer can handle multiple inputs with batching. Let's create another embedding to
        # illustrate that
        embeds = [embed, np.zeros(speaker_embedding_size)]
        texts = ["test 1", "test 2"]
        if self.verbose: print("\tTesting the synthesizer... (loading the model will output a lot of text)")
        mels = self.synthesizer.synthesize_spectrograms(texts, embeds)

        # The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We
        # can concatenate the mel spectrograms to a single one.
        mel = np.concatenate(mels, axis=1)
        # The vocoder can take a callback function to display the generation. More on that later. For
        # now we'll simply hide it like this:
        no_action = lambda *args: None
        if self.verbose: print("\tTesting the vocoder...")
        # For the sake of making this test short, we'll pass a short target length. The target length
        # is the length of the wav segments that are processed in parallel. E.g. for audio sampled
        # at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
        # 0.5 seconds which will all be generated together. The parameters here are absurdly short, and
        # that has a detrimental effect on the quality of the audio. The default parameters are
        # recommended in general.
        vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

        if self.verbose: print("All test passed! You can now synthesize speech.\n\n")
