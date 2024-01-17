"""
Combine whisper transcriptions with PyAnnote diarization

Please see https://github.com/pyannote/pyannote-audio for instructions on accepting their license terms and
place your huggingface access token in a HUGGINGFACE_TOKEN environment variable (or in .env file)
"""
import argparse
import collections
import csv
import logging
from mimetypes import init
import os
import sys

import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils.hook import ProgressHook
from dotenv import load_dotenv

SpeakerSegment = collections.namedtuple(
    "SpeakerSegment", ["start", "end", "speaker_start", "speaker_end", "speaker", "text"]
)


class StderrProgressHook(ProgressHook):
    """Replacement for pyannote progresshook to output to stderr and flush"""

    def __enter__(self):
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            transient=self.transient,
            console=Console(file=sys.stderr),
        )
        self.progress.start()
        return self

    def __call__(self, *args, **kargs):
        try:
            return super().__call__(*args, **kargs)
        finally:
            sys.stderr.flush()


class Diarizer:
    def __init__(self, auth_token=None, whisper_model="large"):
        if auth_token is None:
            load_dotenv()
            auth_token = os.environ.get("HUGGINGFACE_TOKEN")
        if auth_token is None:
            raise Exception(
                "Please provide huggingface auth_token as argument or set HUGGINGFACE_TOKEN environment variable or .env file "
            )
        checkpoint = "pyannote/speaker-diarization-3.1"
        logging.info(f"Loading {checkpoint}")
        self._pipeline = Pipeline.from_pretrained(checkpoint, use_auth_token=auth_token)
        self._pipeline.to(torch.device("cuda"))
        logging.info(f"Loading whisper model {whisper_model}")
        self._whisper = whisper.load_model(whisper_model)
        self._embedding = None
        logging.info(f"Diarization initalized!")

    def _get_embedding(self):
        if self._embedding is None:
            self._embedding = PretrainedSpeakerEmbedding(
                "speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cuda")
            )
        return self._embedding

    def transcribe(self, infile, progress=True, language="nl", initial_prompt=None):
        logging.info(f"Diarizing {infile}")
        with StderrProgressHook() as hook:
            diarization = self._pipeline(infile, hook=hook)

        spreekbeurten = [(turn, speaker) for (turn, _, speaker) in diarization.itertracks(yield_label=True)]

        logging.info(f"Whispering {infile}")
        result = self._whisper.transcribe(infile, language=language, initial_prompt=initial_prompt)

        def guess_speaker(start, end):
            max_d = 0
            best_speaker = None, None
            for turn, speaker in spreekbeurten:
                if turn.start > end:
                    break
                if turn.end < start:
                    continue
                d = min(turn.end, end) - max(turn.start, start)
                if d > max_d:
                    max_d = d
                    best_speaker = (turn, speaker)
            return best_speaker

        for segment in result["segments"]:
            turn, speaker = guess_speaker(segment["start"], segment["end"])
            yield SpeakerSegment(
                segment["start"], segment["end"], turn and turn.start, turn and turn.end, speaker, segment["text"]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename")
    parser.add_argument("--language", default="en")
    parser.add_argument("--prompt", default="en")
    parser.add_argument("--output", choices=["csv", "md"], default="csv")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(name)-12s %(levelname)-5s] %(message)s")
    segments = Diarizer().transcribe(args.filename, language=args.language, initial_prompt=args.prompt)

    if args.output == "csv":
        w = csv.DictWriter(sys.stdout, fieldnames=SpeakerSegment._fields)
        w.writeheader()
        w.writerows((s._asdict() for s in segments))
    elif args.output == "md":
        last_speaker = None
        for s in segments:
            if s.speaker != last_speaker:
                last_speaker = s.speaker
                print(f"\n\n**{s.speaker}**:")
            print(s.text, end=" ")
        print()
