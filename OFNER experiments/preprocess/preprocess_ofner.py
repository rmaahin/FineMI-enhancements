#!/usr/bin/env python3
"""
===============================================================================
 preprocess_ofner.py  —  epoch the OFNER motor-imagery EEG into .npz files
===============================================================================

The OFNER counterpart of FineMI-enhancements/preprocess_code.ipynb. It keeps the
notebook's pipeline order and output format, so the training code reads OFNER
files the same way it reads FineMI files:

    load runs -> concatenate -> average reference -> band-pass (EEG only)
    -> resample to 250 Hz -> epoch on the class triggers -> baseline -> crop

    python preprocess/preprocess_ofner.py --band both               # all 15 subjects
    python preprocess/preprocess_ofner.py --band both --subjects 1,2   # quick check

DATASET
-------
Ofner, Schwarz, Pereira & Muller-Putz (2017, PLOS One), "Upper limb movements can
be decoded from the time-domain of low-frequency EEG"; BNCI Horizon 001-2017,
Zenodo record 834976. Downloaded by download/download_ofner.sh.

  - 15 subjects, motor-imagery session = 10 runs (one GDF file each)
  - 42 trials per run: 6 movements + rest, 60 trials per class per session
  - 512 Hz; 61 EEG + 3 EOG (eog-l/m/r) + 32 misc (glove / Armeo sensors)
  - recorded 0.01-200 Hz, so unlike FineMI there IS signal below 0.5 Hz

TRIAL TIMING (trial time / epoch time)
--------------------------------------
    0 s / -2 s   beep + fixation cross
    2 s /  0 s   cue (the class trigger 1536-1541 is here), sustained imagery
    5 s /  3 s   end of the 3 s task window MOABB uses ("according to paper 2-5")

Epochs are cut -2..3 s around the trigger, baseline-corrected on the fixation
period (-2, 0) - the same idea as the FineMI MIonly data's fixation baseline -
then cropped to 0..3 s. Sample 0 is the cue, so every training window starts at
imagery onset and the fixation period never enters a window.

check_trigger_timing() verifies this from the annotations before anything is
written: the class trigger should sit ~2 s after the trial-start/beep event.

LABELS
------
    1536 elbow flexion   -> 0  EF          1539 pronation   -> 3  PRO
    1537 elbow extension -> 1  EE          1540 hand close  -> 4  HC
    1538 supination      -> 2  SUP         1541 hand open   -> 5  HO
    1542 rest            -> not epoched

The notebook's extract_trigger_num() regex only accepts 1..8, so it would drop
every OFNER code. Labels here come from an explicit code -> label map instead.

OUTPUT
------
One folder per band, nothing else touched:

    <out-root>/ofner_mi_0.5_3hz/subject{N}_eeg_epochs_0.5_3hz_250hz.npz
    <out-root>/ofner_mi_8_30hz/subject{N}_eeg_epochs_8_30hz_250hz.npz

    data      (360, 61, 750)  float64, volts, 0..3 s after the cue at 250 Hz
    labels    (360,)          0..5 as above
    onsets    (360,)          trigger sample in the 250 Hz concatenated raw
    ch_names  (61,)

Separate folders per band for the same reason as the notebook: the loader globs
subject{N}_eeg_epochs_<band>_*.npz, and a second band in the same folder would be
a second match.

DECIMATION
----------
Files are written at 250 Hz, like FineMI. The training loader then keeps every
5th sample for 0.5-3 Hz (50 Hz, Nyquist 25 Hz) and every sample for 8-30 Hz:
50 Hz would alias 25-30 Hz, so that band stays at 250 Hz.
===============================================================================
"""
import argparse
import gc
import glob
import os
import re
import sys

import numpy as np

try:
    import mne
except ImportError:
    sys.exit("MNE is not installed in this environment: pip install mne")

mne.set_log_level('ERROR')
print(f"MNE {mne.__version__} loaded OK  (python {sys.version.split()[0]})")


# ==========================================================================
#  CONFIG — defaults for the command-line flags
# ==========================================================================
_DEFAULT_DATA_ROOT = (os.path.join(os.environ['SCRATCH'], 'ofner')
                      if os.environ.get('SCRATCH') else '/workspace/ofner')
RAW = os.path.join(_DEFAULT_DATA_ROOT, 'raw')   # folder with motorimagination_*.gdf
OUT_ROOT = _DEFAULT_DATA_ROOT                    # ofner_mi_<band>/ folders go here
SUBJECTS = ''         # '' = all 15; or e.g. '1,2' to test on a few first
FORCE = False         # True = recompute subjects already written

BANDS = {             # --band choice -> (l_freq, h_freq)
    '0.5_3': (0.5, 3.0),
    '8_30': (8.0, 30.0),
}
TARGET_SFREQ = 250.0
EPOCH_TMIN, EPOCH_TMAX = -2.0, 3.0   # around the cue; -2 s = beep + fixation cross
BASELINE = (-2.0, 0.0)               # fixation period
CROP_TMIN, CROP_TMAX = 0.0, 3.0 - 1 / TARGET_SFREQ   # 750 samples
N_EEG = 61
N_RUNS = 10
EXPECT_TRIALS = 360                  # 6 movement classes x 60
# ==========================================================================


# Map trigger codes to labels 0-5 and task names. 1542 (rest) is deliberately absent.
CODE_TO_LABEL = {1536: 0, 1537: 1, 1538: 2, 1539: 3, 1540: 4, 1541: 5}
CLASS_NAMES = ('EF', 'EE', 'SUP', 'PRO', 'HC', 'HO')
MI_TASKS = {
    0: "elbow_flexion",
    1: "elbow_extension",
    2: "forearm_supination",
    3: "forearm_pronation",
    4: "hand_close",
    5: "hand_open",
}
REST_CODE = 1542


# =============================================================================
# Helper utilities
# =============================================================================

def run_files(raw_dir, subject):
    """The subject's 10 motor-imagery GDF files in run order (missing ones omitted)."""
    files = [os.path.join(raw_dir, f"motorimagination_subject{subject}_run{r}.gdf")
             for r in range(1, N_RUNS + 1)]
    return [f for f in files if os.path.exists(f)]


def available_subjects(raw_dir):
    subs = set()
    for f in glob.glob(os.path.join(raw_dir, "motorimagination_subject*_run*.gdf")):
        m = re.search(r"subject(\d+)_run", os.path.basename(f))
        if m:
            subs.add(int(m.group(1)))
    return sorted(subs)


def _channel_kinds(ch_names):
    """Split channel names into (eeg, eog, misc) by OFNER's layout.

    Channels 0-63 are 61 EEG + 3 EOG; 64-95 are glove / Armeo sensors. EOG is
    recognized by name because subject 1's runs order it differently, and subject
    1's imagery runs carry 'armeodummy-*' placeholders that are misc wherever they sit.
    """
    eeg, eog, misc = [], [], []
    for i, ch in enumerate(ch_names):
        low = ch.lower()
        if low.startswith('eog'):
            eog.append(ch)
        elif i >= 64 or low.startswith('armeo'):
            misc.append(ch)
        else:
            eeg.append(ch)
    return eeg, eog, misc


def _is_generic_eeg_name(ch):
    return re.fullmatch(r"eeg-\d+", ch.lower()) is not None


def reference_eeg_names(raw_dir, exclude_subject):
    """The 61 EEG channel names, in order, read from another subject's run-1 header.

    Used to rename subject 1's generic 'eeg-0'..'eeg-60' channels by position.
    """
    for s in available_subjects(raw_dir):
        if s == exclude_subject:
            continue
        files = run_files(raw_dir, s)
        if not files:
            continue
        hdr = mne.io.read_raw_gdf(files[0], preload=False, verbose=False)
        eeg, _, _ = _channel_kinds(hdr.ch_names)
        if len(eeg) == N_EEG and not any(_is_generic_eeg_name(c) for c in eeg):
            return eeg, s
    raise RuntimeError(
        "subject has generic eeg-* channel names, and no other subject's run-1 file "
        f"in {raw_dir} has {N_EEG} named EEG channels to copy them from. "
        "Download at least one more subject.")


def _safe_print_annotations(raw, n=10):
    """Print a small annotation preview safely."""
    if raw.annotations is not None and len(raw.annotations) > 0:
        print("Annotation descriptions (first):", raw.annotations.description[:n])
        print("Annotation onsets (first):      ", np.round(raw.annotations.onset[:n], 3))
    else:
        print("No annotations present in raw.")


def _code_of(description):
    """GDF annotation description -> integer event code, or None ('1536' -> 1536)."""
    s = str(description).strip()
    return int(s) if s.isdigit() else None


def check_trigger_timing(raw, verbose=True):
    """Check that the class triggers mark the CUE, not the start of the trial.

    For every other event code (trial start, beep, cross, ...), take its offset to
    the nearest class/rest trigger within +-3 s, and the median over occurrences:

      a code at ~-2 s  -> it is the beep 2 s BEFORE the trigger: trigger = cue, OK
      a code at ~+2 s  -> the trigger is 2 s before the cue = trial start: abort,
                          the epoch must move by +2 s
      neither          -> cannot verify from the file; warn and continue

    Returns {code: (median offset s, n)} for the other codes.
    """
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    descs = np.asarray(raw.annotations.description)
    codes = [_code_of(d) for d in descs]

    task_codes = set(CODE_TO_LABEL) | {REST_CODE}
    task_onsets = np.sort(onsets[[c in task_codes for c in codes]])

    found = {}
    for d in descs:
        found[str(d)] = found.get(str(d), 0) + 1

    offsets = {}                      # other code -> [onset - nearest trigger, ...]
    for t, c in zip(onsets, codes):
        if c is None or c in task_codes or not task_onsets.size:
            continue
        d = t - task_onsets[np.argmin(np.abs(task_onsets - t))]
        if abs(d) <= 3.0:
            offsets.setdefault(c, []).append(d)
    summary = {c: (float(np.median(v)), len(v)) for c, v in offsets.items()}

    if verbose:
        print("  annotation codes (description: count):")
        for d in sorted(found, key=lambda x: (_code_of(x) is None, _code_of(x) or 0, x)):
            tag = ''
            c = _code_of(d)
            if c in CODE_TO_LABEL:
                tag = f"  <- {CLASS_NAMES[CODE_TO_LABEL[c]]}"
            elif c == REST_CODE:
                tag = "  <- rest (not epoched)"
            print(f"    {d:>14}: {found[d]}{tag}")
        for c, (med, n) in sorted(summary.items()):
            print(f"  code {c}: median offset {med:+.3f} s from the nearest trigger (n={n})")

    before = [c for c, (med, _) in summary.items() if -2.5 <= med <= -1.5]
    after = [c for c, (med, _) in summary.items() if 1.5 <= med <= 2.5]
    if after and not before:
        raise RuntimeError(
            f"code(s) {after} sit ~2 s AFTER the class triggers, so the triggers mark "
            "TRIAL START, not the cue. The epoch window is then 2 s too early: shift "
            "EPOCH_TMIN/EPOCH_TMAX/BASELINE/CROP_* by +2 s before using this data.")
    if before:
        print(f"  timing check OK: code(s) {before} sit ~2 s before the triggers "
              "(beep / fixation cross), so trigger = cue.")
    else:
        print("  !! timing check: no event code sits ~2 s from the triggers, so the cue "
              "position cannot be verified from this file. Assuming trigger = cue, as "
              "MOABB does (interval [0, 3] = the paper's 2-5 s).")
    return summary


# =============================================================================
# Core preprocessing functions
# =============================================================================

def load_subject_eeg(raw_dir, subject, ref_names=None, verbose=True):
    """
    Load the subject's 10 motor-imagery runs as ONE referenced raw (not yet filtered):
    - reads each GDF, keeps the 61 EEG channels (EOG and sensors dropped)
    - subject 1: renames generic eeg-* channels by position from ref_names
    - NaN -> 0, microvolts -> volts (as MOABB's Ofner2017 loader does)
    - concatenates the runs in order
    - sets montage (standard_1005, ignore missing)
    - average reference (applied)
    The band-pass and resampling happen per band in filter_and_resample().
    """
    files = run_files(raw_dir, subject)
    print(f"  subject{subject}: {len(files)} run file(s)"
          + (f"   <-- FEWER THAN {N_RUNS}, check the download" if len(files) < N_RUNS else ""))
    if not files:
        raise FileNotFoundError(f"no motorimagination_subject{subject}_run*.gdf in {raw_dir}")

    all_runs = []
    eeg_order = None
    for path in files:
        if verbose:
            print(f"    Loading {os.path.basename(path)}...")
        try:
            run = mne.io.read_raw_gdf(path, preload=True, verbose=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

        eeg, eog, misc = _channel_kinds(run.ch_names)
        if len(eeg) != N_EEG:
            raise RuntimeError(f"{os.path.basename(path)}: {len(eeg)} EEG channels, "
                               f"expected {N_EEG} (eog={eog}, {len(misc)} misc)")

        # Keep EEG only. EOG and sensors are never used, and dropping them first
        # sidesteps subject 1's different EOG order / Armeo placeholders.
        run.pick(eeg)
        run.set_channel_types({ch: 'eeg' for ch in run.ch_names}, verbose=False)

        generic = [c for c in run.ch_names if _is_generic_eeg_name(c)]
        if generic:
            if ref_names is None or len(generic) != len(run.ch_names):
                raise RuntimeError(f"{os.path.basename(path)}: generic channel names "
                                   f"and no {N_EEG}-name reference to rename them from")
            run.rename_channels(dict(zip(run.ch_names, ref_names)))
            if verbose:
                print(f"      renamed {len(generic)} generic eeg-* channels by position")

        # same channel order in every run, or concatenate_raws refuses
        if eeg_order is None:
            eeg_order = list(run.ch_names)
        elif list(run.ch_names) != eeg_order:
            if sorted(run.ch_names) != sorted(eeg_order):
                raise RuntimeError(f"{os.path.basename(path)}: EEG channels differ from run 1")
            run.reorder_channels(eeg_order)

        data = run._data
        n_nan = int(np.isnan(data).sum())
        if n_nan:
            data[np.isnan(data)] = 0.0
            if verbose:
                print(f"      {n_nan} NaN samples set to 0")
        data *= 1e-6                    # file values are microvolts

        all_runs.append(run)

    sfreqs = {r.info["sfreq"] for r in all_runs}
    if len(sfreqs) > 1:
        raise RuntimeError(f"Runs have different sampling rates: {sfreqs}")

    raw = all_runs[0] if len(all_runs) == 1 else mne.concatenate_raws(all_runs, preload=True,
                                                                      verbose=False)

    try:
        raw.set_montage("standard_1005", match_case=False, on_missing="ignore", verbose=False)
    except Exception:
        if verbose:
            print("    Warning: Could not set montage (standard_1005). Continuing...")

    # Apply average reference BEFORE filtering (as in the notebook)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    if verbose:
        amp = float(np.median(np.abs(raw.get_data()))) * 1e6
        print(f"  ✓ Loaded: {len(raw.ch_names)} EEG channels, {raw.info['sfreq']:.0f} Hz, "
              f"{raw.times[-1]:.1f} s, median |x| = {amp:.2f} uV")
        _safe_print_annotations(raw, n=10)

    return raw


def filter_and_resample(raw, l_freq, h_freq, target_sampling_rate=TARGET_SFREQ, verbose=True):
    """Band-pass a COPY of the referenced raw on EEG channels, then resample.

    raw.filter skips across the EDGE boundaries that concatenate_raws inserts, so each
    run is filtered on its own and no filter transient crosses a run boundary.
    """
    out = raw.copy()
    if verbose:
        print(f"  Applying bandpass filter ({l_freq}-{h_freq} Hz) on EEG channels...")
    out.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)

    current_rate = float(out.info["sfreq"])
    if abs(current_rate - float(target_sampling_rate)) > 0.1:
        if verbose:
            print(f"  Resampling from {current_rate:.2f} Hz to {target_sampling_rate:.2f} Hz...")
        out.resample(target_sampling_rate, verbose=False)
    return out


def extract_epochs(raw, verbose=True):
    """
    Epoch on the six movement triggers, baseline on the fixation period, crop to
    0..3 s after the cue. Returns (epochs, labels 0..5).
    """
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    if events is None or len(events) == 0:
        raise ValueError("No events found in the annotations.")

    # event_id: description -> MNE code, for the six movement codes only
    event_id = {d: c for d, c in event_dict.items() if _code_of(d) in CODE_TO_LABEL}
    if len(event_id) != len(CODE_TO_LABEL):
        missing = sorted(set(CODE_TO_LABEL) - {_code_of(d) for d in event_id})
        raise ValueError(f"movement codes missing from the annotations: {missing}\n"
                         f"Event names found: {sorted(event_dict)}")
    mne_code_to_label = {c: CODE_TO_LABEL[_code_of(d)] for d, c in event_id.items()}

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=BASELINE,
        picks="eeg", reject_by_annotation=True, preload=True, verbose=False)
    if len(epochs) == 0:
        raise ValueError("No epochs were created. Check event timing and tmin/tmax range.")

    epochs.crop(tmin=CROP_TMIN, tmax=CROP_TMAX)
    labels = np.array([mne_code_to_label[int(c)] for c in epochs.events[:, -1]], dtype=int)

    if verbose:
        print(f"  ✓ Extracted epochs: {epochs.get_data().shape} = (trials, channels, timepoints)")
        if len(epochs.drop_log) and any(epochs.drop_log):
            n_drop = sum(1 for d in epochs.drop_log if d)
            print(f"    {n_drop} epoch(s) dropped (overlapping a BAD/EDGE annotation)")
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("    Label distribution:")
        for lab, cnt in zip(unique_labels, counts):
            print(f"      Label {lab} ({CLASS_NAMES[lab]}, {MI_TASKS[int(lab)]}): {cnt}")

    return epochs, labels


def save_epochs_numpy(epochs, labels, out_path):
    """Save epochs and labels to compressed NumPy file (notebook format + ch_names)."""
    data = epochs.get_data()
    labels = np.asarray(labels, dtype=int)
    trial_onsets = epochs.events[:, 0].astype(int)  # sample index in (resampled) raw
    np.savez_compressed(out_path, data=data, labels=labels, onsets=trial_onsets,
                        ch_names=np.array(epochs.ch_names))
    print(f"  Saved: {out_path}")
    print(f"    data shape: {data.shape}   labels: {labels.shape}   onsets: {trial_onsets.shape}")


def band_power_fraction(data, sfreq, l_freq, h_freq):
    """Fraction of the epochs' power (Welch, channel-averaged, DC bin excluded) inside
    the filter's -6 dB band. MNE's default FIR transitions are wide at low cutoffs
    (0.5 Hz -> 0.25 Hz at -6 dB, 3 Hz -> 4 Hz), and EEG's 1/f spectrum piles power
    at the lower edge, so counting only [l_freq, h_freq] flags a correct 0.5-3 Hz filter."""
    from scipy.signal import welch
    l_trans = min(max(0.25 * l_freq, 2.0), l_freq)              # MNE 'auto' widths
    h_trans = min(max(0.25 * h_freq, 2.0), sfreq / 2.0 - h_freq)
    f, p = welch(data, fs=sfreq, nperseg=data.shape[-1], axis=-1)
    p = p.mean(axis=(0, 1))[1:]
    f = f[1:]
    inside = (f >= l_freq - l_trans / 2) & (f <= h_freq + h_trans / 2)
    return float(p[inside].sum() / p.sum()), float(f[np.argmax(p)])


# =============================================================================
#  MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Epoch the OFNER motor-imagery EEG into .npz files.")
    ap.add_argument('--raw', default=RAW,
                    help='folder with motorimagination_subject*_run*.gdf (download_ofner.sh)')
    ap.add_argument('--out-root', default=OUT_ROOT,
                    help='ofner_mi_<band>/ output folders are created here')
    ap.add_argument('--band', choices=list(BANDS) + ['both'], default='both')
    ap.add_argument('--subjects', default=SUBJECTS,
                    help='comma-separated subject numbers (default: all found)')
    ap.add_argument('--expect_trials', type=int, default=EXPECT_TRIALS,
                    help='movement trials per subject (6 classes x 60 = 360). '
                         'Warns if a subject comes up short.')
    ap.add_argument('--force', action='store_true', default=FORCE,
                    help='recompute subjects whose output already exists')
    a = ap.parse_args()

    bands = list(BANDS) if a.band == 'both' else [a.band]
    band_tags = {b: f"{BANDS[b][0]:g}_{BANDS[b][1]:g}hz" for b in bands}
    out_dirs = {b: os.path.join(a.out_root, f"ofner_mi_{band_tags[b]}") for b in bands}

    subs = available_subjects(a.raw)
    if a.subjects.strip():
        want = {int(s) for s in a.subjects.split(',') if s.strip()}
        subs = [s for s in subs if s in want]

    print("=" * 74)
    print(f"bands     : " + ", ".join(f"{BANDS[b][0]}-{BANDS[b][1]} Hz" for b in bands))
    print(f"raw       : {a.raw}")
    for b in bands:
        print(f"output    : {out_dirs[b]}")
    print(f"subjects  : {subs}")
    print(f"epochs    : {EPOCH_TMIN}..{EPOCH_TMAX} s around the cue, baseline {BASELINE}, "
          f"cropped to {CROP_TMIN}..{CROP_TMAX:.3f} s at {TARGET_SFREQ:g} Hz")
    print("=" * 74)
    if not subs:
        print("No motorimagination_subject*_run*.gdf files found - check --raw"); return 1

    # Fail now, with a fix, rather than after an hour of filtering.
    for d in out_dirs.values():
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, '.write_test')
            with open(probe, 'w') as f:
                f.write('ok')
            os.remove(probe)
        except OSError as e:
            print(f"CANNOT WRITE TO: {d}\n  {type(e).__name__}: {e}\nSet --out-root.")
            return 1

    def out_file(b, s):
        return os.path.join(out_dirs[b], f"subject{s}_eeg_epochs_{band_tags[b]}_250hz.npz")

    done, failed = 0, []
    for s in subs:
        todo = [b for b in bands if a.force or not os.path.exists(out_file(b, s))]
        if not todo:
            print(f"subject{s:<4} already done, skipping")
            done += 1
            continue
        print(f"\n=== subject{s} ===", flush=True)
        raw = None
        try:
            ref_names = None
            hdr = mne.io.read_raw_gdf(run_files(a.raw, s)[0], preload=False, verbose=False)
            if any(_is_generic_eeg_name(c) for c in _channel_kinds(hdr.ch_names)[0]):
                ref_names, ref_s = reference_eeg_names(a.raw, exclude_subject=s)
                print(f"  generic eeg-* names: renaming by position from subject{ref_s}")
            del hdr

            raw = load_subject_eeg(a.raw, s, ref_names=ref_names, verbose=True)
            check_trigger_timing(raw, verbose=True)

            for b in todo:
                l_freq, h_freq = BANDS[b]
                print(f"  --- {l_freq}-{h_freq} Hz ---")
                raw_b = filter_and_resample(raw, l_freq, h_freq, verbose=True)
                epochs, labels = extract_epochs(raw_b, verbose=True)
                n_tr = len(labels)
                if n_tr < a.expect_trials:
                    print(f"  !! {n_tr} trials, expected {a.expect_trials} "
                          f"({a.expect_trials - n_tr} missing) — check the runs above.")
                save_epochs_numpy(epochs, labels, out_file(b, s))
                del raw_b, epochs, labels
                gc.collect()
            done += 1
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(s)
        finally:
            del raw
            gc.collect()

    print(f"\n{'=' * 74}\n{done} subject(s) written, {len(failed)} failed {failed}")

    # ---------------------------------------------------------------------
    #  VERIFY. Same trials in every band, the expected shapes, and each band's
    #  power really inside its passband.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("VERIFICATION")
    print("=" * 74)
    problems = []
    hdr = f"{'subject':<10}{'band':>10}{'shape':>18}{'per class':>26}{'in-band':>9}{'peak Hz':>9}"
    print(hdr)
    loaded = {}
    for s in subs:
        for b in bands:
            path = out_file(b, s)
            if not os.path.exists(path):
                print(f"subject{s:<3}{band_tags[b]:>10}   file missing")
                problems.append(f"subject{s} {b}")
                continue
            with np.load(path, allow_pickle=True) as z:
                x, y, on = z['data'], z['labels'], z['onsets']
            counts = np.bincount(y, minlength=len(CLASS_NAMES))
            frac, peak = band_power_fraction(x, TARGET_SFREQ, *BANDS[b])
            bad = (x.shape[1:] != (N_EEG, 750) or len(y) != a.expect_trials
                   or not np.isfinite(x).all() or (counts != a.expect_trials // 6).any()
                   or frac < 0.7)
            print(f"subject{s:<3}{band_tags[b]:>10}{str(x.shape):>18}"
                  f"{' '.join(str(c) for c in counts):>26}{frac:>9.2f}{peak:>9.2f}"
                  + ("   <-- CHECK" if bad else ""))
            if bad:
                problems.append(f"subject{s} {b}")
            loaded[(s, b)] = (x, y, on)

        if len(bands) == 2 and all((s, b) in loaded for b in bands):
            (x0, y0, on0), (x1, y1, on1) = loaded[(s, bands[0])], loaded[(s, bands[1])]
            same = (y0.shape == y1.shape and (y0 == y1).all() and (on0 == on1).all())
            c = float(np.corrcoef(x0[:20].ravel(), x1[:20].ravel())[0, 1])
            ok = same and abs(c) < 0.5
            print(f"{'':<10}{'cross':>10}  trials {'identical' if same else 'DIFFER'}, "
                  f"corr {c:.3f}" + ("" if ok else "   <-- CHECK"))
            if not ok:
                problems.append(f"subject{s} cross-band")
        for b in bands:
            loaded.pop((s, b), None)

    print()
    if problems or failed:
        print(f"PROBLEM with: {problems + [f'subject{s} failed' for s in failed]}")
        print("  shape / per class -> runs missing or epochs dropped; do not use that subject")
        print("  in-band < 0.70    -> most power is outside the passband; filter did not apply")
        print("  trials DIFFER     -> the two bands hold different trials; comparison broken")
        print("  |corr| > 0.5      -> the two bands look alike; band arguments did not apply")
        return 1
    print("ALL GOOD - expected shapes, 60 trials per class, power inside each passband"
          + (", same trials in both bands." if len(bands) == 2 else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
