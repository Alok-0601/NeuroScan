"""
slim_model.py — remove unused optimiser state from the saved model
==================================================================

Run this ONCE, locally, then commit the result.

WHY THIS EXISTS
---------------
`brain_tumor_vgg16.keras` is 147 MiB, but the network itself only accounts for
68 MiB of that:

    model weights      17,926,596 params x 4 bytes  =  68.4 MiB
    Adam state         20,582,664 slots  x 4 bytes  =  78.5 MiB
                                                      ---------
                                                        146.9 MiB

The second row is the optimiser's bookkeeping — one momentum buffer and one
variance buffer for each of the 10,291,332 parameters that were trainable during
fine-tuning (the last three VGG16 conv layers plus the classifier head). Keras
stores it so that training can be *resumed* mid-run. Inference never reads it.

Dropping it matters for two practical reasons:

  1. 68 MiB is under GitHub's 100 MB per-file limit, so the model can be
     committed as an ordinary file. No Git LFS. This is the important one —
     Streamlit Community Cloud does not resolve LFS objects, so an LFS-tracked
     model arrives at the host as a ~130-byte pointer file and fails to load.

  2. Less to download and less to hold in memory on a memory-capped host.

WHAT THIS DOES NOT DO
---------------------
It does not retrain, fine-tune, re-architect, or alter a single weight. It loads
the model with `compile=False` (which skips restoring the optimiser) and saves it
again. Every weight array is copied across untouched — and the script proves that
by comparing all 32 weight arrays element-for-element and diffing predictions
before it replaces anything. If verification fails, nothing is swapped.

USAGE
-----
    python slim_model.py

Your original file is kept as brain_tumor_vgg16.keras.bak (gitignored) so you can
always roll back.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import zipfile
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "brain_tumor_vgg16.keras"
SEARCH_DIRS = [".", "Models", "models", "model", "saved_model", "artifacts"]

IMAGE_SIZE = (224, 224)
MIB = 1024 * 1024


def find_model():
    """Locate the model file, matching the search order app.py uses."""
    for folder in SEARCH_DIRS:
        candidate = APP_DIR / folder / MODEL_FILENAME
        if candidate.is_file():
            return candidate
    return None


def has_optimizer_state(keras_path):
    """Report whether the .keras archive carries optimiser variables.

    A .keras file is just a zip containing config.json, metadata.json and
    model.weights.h5. HDF5 writes group names as plain ASCII, so the presence of
    an 'optimizer' group can be checked without TensorFlow or h5py installed.
    """
    with zipfile.ZipFile(keras_path) as archive:
        if "model.weights.h5" not in archive.namelist():
            return False
        blob = archive.read("model.weights.h5")
    return b"optimizer" in blob.lower()


def main():
    source = find_model()
    if source is None:
        print(f"ERROR: could not find {MODEL_FILENAME}.")
        print(f"       Looked in: {', '.join(SEARCH_DIRS)} (relative to this script)")
        return 1

    original_mib = source.stat().st_size / MIB
    print(f"Found  {source}")
    print(f"Size   {original_mib:.1f} MiB")

    if not has_optimizer_state(source):
        print("\nThis file has no optimiser state — it is already slim.")
        print("Nothing to do.")
        return 0

    print("Optimiser state present. Proceeding.\n")

    # Imported here rather than at module top so the "already slim" and
    # "not found" paths stay fast and do not need TensorFlow at all.
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow is not installed in this environment.")
        print("       Run this in the same env you trained the model in, or:")
        print('       pip install "tensorflow-cpu>=2.16" "keras>=3.13.2"')
        return 1

    print(f"TensorFlow {tf.__version__} / Keras {tf.keras.__version__}")

    # -----------------------------------------------------------------------
    # Re-save without the optimiser
    # -----------------------------------------------------------------------
    print("\nLoading original (compile=False skips the optimiser)...")
    slim = tf.keras.models.load_model(source, compile=False)

    destination = source.with_name(source.stem + ".slim.keras")
    print(f"Saving  {destination.name} ...")
    slim.save(destination)

    slim_mib = destination.stat().st_size / MIB
    saved = original_mib - slim_mib
    print(f"\n  before  {original_mib:7.1f} MiB")
    print(f"  after   {slim_mib:7.1f} MiB")
    print(f"  saved   {saved:7.1f} MiB  ({saved / original_mib * 100:.0f}% smaller)")

    if slim_mib >= 100:
        print("\nWARNING: still 100 MiB or more, so GitHub will reject it.")
        print("         Not swapping. Investigate before committing.")
        return 1

    # -----------------------------------------------------------------------
    # Verify nothing changed
    # -----------------------------------------------------------------------
    print("\n" + "-" * 62)
    print("VERIFYING the weights survived untouched")
    print("-" * 62)

    original = tf.keras.models.load_model(source, compile=False)
    reloaded = tf.keras.models.load_model(destination, compile=False)

    before, after = original.get_weights(), reloaded.get_weights()

    if len(before) != len(after):
        print(f"FAIL: array count differs ({len(before)} vs {len(after)}).")
        return 1

    worst = 0.0
    for index, (a, b) in enumerate(zip(before, after)):
        if a.shape != b.shape:
            print(f"FAIL: array {index} shape {a.shape} vs {b.shape}.")
            return 1
        worst = max(worst, float(np.abs(a - b).max()))

    total_params = sum(int(a.size) for a in before)
    print(f"  weight arrays compared   {len(before)}")
    print(f"  parameters compared      {total_params:,}")
    print(f"  largest difference       {worst:.3e}")

    if worst != 0.0:
        print("\nFAIL: weights are not bit-identical. Not swapping.")
        return 1
    print("  -> every parameter identical")

    # Predictions on fixed pseudo-random input. Dropout is inactive during
    # inference, so this is deterministic and any drift would show up here.
    rng = np.random.default_rng(0)
    probe = rng.uniform(-124, 151, size=(4, *IMAGE_SIZE, 3)).astype("float32")
    delta = float(np.abs(original.predict(probe, verbose=0)
                         - reloaded.predict(probe, verbose=0)).max())
    print(f"  prediction delta         {delta:.3e}")
    if delta != 0.0:
        print("\nFAIL: predictions differ. Not swapping.")
        return 1
    print("  -> predictions identical")

    # And once on a real scan, so the check covers the true preprocessing path.
    samples = sorted((APP_DIR / "samples").glob("*.jpg")) if (APP_DIR / "samples").is_dir() else []
    if samples:
        from tensorflow.keras.applications.vgg16 import preprocess_input

        scan = tf.keras.utils.load_img(samples[0], target_size=IMAGE_SIZE, color_mode="rgb")
        batch = preprocess_input(np.expand_dims(tf.keras.utils.img_to_array(scan), axis=0))
        probabilities = reloaded.predict(batch, verbose=0)[0]
        print(f"  real scan ({samples[0].name}): "
              f"top class index {int(np.argmax(probabilities))} "
              f"at {float(np.max(probabilities)) * 100:.2f}%")

    # -----------------------------------------------------------------------
    # Swap, keeping a backup
    # -----------------------------------------------------------------------
    backup = source.with_suffix(source.suffix + ".bak")
    print("\n" + "-" * 62)
    print(f"Verification passed. Swapping files.")
    print(f"  {source.name} -> {backup.name}  (backup, gitignored)")
    if backup.exists():
        backup.unlink()
    source.rename(backup)
    destination.rename(source)
    print(f"  {destination.name} -> {source.name}")

    print(f"\nDone. {source.name} is now {slim_mib:.1f} MiB.\n")
    relative = source.relative_to(APP_DIR).as_posix()
    print("Commit it as a normal file — no Git LFS needed:")
    print(f"    git rm --cached {relative}      # only if it was LFS-tracked before")
    print(f"    git add {relative}")
    print('    git commit -m "Store model without optimiser state (147 -> 68 MiB)"')
    print("    git push")
    print("\nGitHub warns above 50 MiB but only blocks above 100 MiB, so expect a")
    print("warning on push. It will succeed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
