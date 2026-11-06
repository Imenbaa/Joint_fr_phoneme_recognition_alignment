#!/usr/bin/env python3
"""Build an evaluation manifest (see evaluate.py) from audio files and Praat
TextGrids with a phone tier.

For every <audio_dir>/**/<id>.wav with a TextGrid <textgrid_dir>/<id><suffix>:
  chunks              the pieces JointRecAlign transcribe cuts the recording into
                      (jointrecalign/transcribe.py, chunk_bounds): at most --max_chunk_dur
                      seconds, back to back from 0 to the end, each cut in the middle
                      of a pause found by Silero VAD (or, with no pause, where speech
                      is least likely), so the evaluation scores what the toolkit does
  ref_intervals       the phone intervals of --tier, labels mapped with --phone_map;
                      pause and skipped labels are left out
  pause_intervals     intervals whose label is in --pause_labels (used by the SIL scores)
  excluded_intervals  intervals whose label is in --skip_labels (non-phone labels
                      that are not pauses: noise, overlaps, unknown words, ...)
  style               optional grouping key, from --style rules on the file id

Runs in the same environment as the rest (requirements.txt):

    python make_manifest.py --audio_dir wav/ --textgrid_dir textgrids/ --tier phones \\
        --phone_map phone_map.json --out manifest.json

--phone_map is a JSON object {"tier label": "vocabulary phone"} that rewrites the
reference labels into the model's phone set (vocab.json); unmapped labels are
kept as they are.
"""
import argparse
import glob
import json
import os

from jointrecalign.evaluation.attention_eval import SR, load_audio
from jointrecalign.reference import read_textgrid as read_tiers
from jointrecalign.transcribe import chunk_bounds

PAUSE_LABELS = ["", "SIL", "sil", "spn", "SP", "<SIL>", "_"]


def read_textgrid(path, tier, phone_map, pause_labels, skip_labels):
    """-> (ref_intervals, pause_intervals, excluded_intervals)"""
    tiers = read_tiers(path)
    if tier not in tiers:
        raise SystemExit(f"{path}: no interval tier {tier!r} (tiers: {', '.join(tiers) or 'none'})")
    ref, pauses, excluded = [], [], []
    for start, end, text in tiers[tier]:
        lab = text.strip()
        s, e = round(start, 6), round(end, 6)
        if lab in pause_labels:
            pauses.append([s, e])
            continue
        if lab in skip_labels:
            excluded.append([s, e, lab])
            continue
        ph = phone_map.get(lab, lab)
        if ph in pause_labels or ph in skip_labels:
            continue
        ref.append({"phoneme": ph, "start": s, "end": e})
    return ref, pauses, excluded


def style_of(file_id, rules):
    for pattern, style in rules:
        if pattern in file_id:
            return style
    return "ALL"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio_dir", required=True, help="searched recursively for *.wav")
    p.add_argument("--textgrid_dir", required=True)
    p.add_argument("--textgrid_suffix", default=".TextGrid",
                   help="TextGrid file name = <audio id> + this suffix "
                        "(write --textgrid_suffix=-x.TextGrid if it starts with '-')")
    p.add_argument("--tier", required=True, help="phone tier name")
    p.add_argument("--out", required=True)
    p.add_argument("--phone_map", default=None, help="JSON {tier label: vocabulary phone}")
    p.add_argument("--pause_labels", nargs="*", default=PAUSE_LABELS,
                   help="labels of pause intervals (default: %(default)s)")
    p.add_argument("--skip_labels", nargs="*", default=[],
                   help="other non-phone labels to leave out of the reference")
    p.add_argument("--style", default="", metavar="SUBSTRING=STYLE,...",
                   help="style of a file whose id contains SUBSTRING (first match), "
                        "e.g. --style=_read=read,_spont=spontaneous")
    p.add_argument("--max_chunk_dur", type=float, default=10.0,
                   help="longest chunk in seconds (default 10, as JointRecAlign transcribe)")
    p.add_argument("--skip", nargs="*", default=[], help="file ids to leave out")
    a = p.parse_args()

    phone_map = json.load(open(a.phone_map)) if a.phone_map else {}
    pause_labels, skip_labels = set(a.pause_labels), set(a.skip_labels)
    rules = [r.split("=", 1) for r in a.style.split(",") if r]

    man = {"config": vars(a), "files": {}}
    wavs = sorted(glob.glob(os.path.join(a.audio_dir, "**", "*.wav"), recursive=True))
    for k, wp in enumerate(wavs):
        nm = os.path.splitext(os.path.basename(wp))[0]
        if nm in a.skip:
            continue
        tg = os.path.join(a.textgrid_dir, nm + a.textgrid_suffix)
        if not os.path.exists(tg):
            print(f"[skip] no TextGrid: {nm}")
            continue
        audio = load_audio(wp)
        ref, pauses, excluded = read_textgrid(tg, a.tier, phone_map, pause_labels, skip_labels)
        chunks = [{"start": float(s), "end": float(e)}
                  for s, e in chunk_bounds(audio, SR, max_chunk=a.max_chunk_dur)]
        man["files"][nm] = {
            "wav": os.path.abspath(wp), "duration": len(audio) / SR,
            "style": style_of(nm, rules),
            "ref_intervals": ref,
            "chunks": chunks,
            "pause_intervals": pauses,
            "excluded_intervals": excluded,
        }
        print(f"[{k + 1}/{len(wavs)}] {nm}  ref={len(ref)}  pauses={len(pauses)}  chunks={len(chunks)}")

    json.dump(man, open(a.out, "w"), ensure_ascii=False)
    print(f"\n[saved] {a.out}  {len(man['files'])} files, "
          f"{sum(len(v['chunks']) for v in man['files'].values())} chunks")


if __name__ == "__main__":
    main()
