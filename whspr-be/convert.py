import os, shutil

# === FINAL TARGET ===
dst = "train_data"

# === CREMA-D SOURCE ===
cremad_src = "cremad/AudioWAV"

# === TESS SOURCE ===
tess_src = "tess"

# === RAVDESS SOURCE (already good) ===
ravdess_src = "Ravdess"

# === FINAL LABEL MAP ===
MAP = {
    # CREMA-D
    'ANG': 'angry',
    'HAP': 'happy',
    'SAD': 'sad',
    'NEU': 'neutral',
    'FEA': 'frustrated',
    'DIS': 'frustrated',

    # TESS
    'angry': 'angry',
    'disgust': 'frustrated',
    'fear': 'frustrated',
    'happy': 'happy',
    'neutral': 'neutral',
    'pleasant_surprise': 'satisfied',
    'pleasant_surprised': 'satisfied',
    'sad': 'sad',
}

# ensure folders exist
def ensure(label):
    os.makedirs(f"{dst}/{label}", exist_ok=True)

# =========================
# 1. COPY RAVDESS (already correct)
# =========================
for folder in os.listdir(ravdess_src):
    src_folder = os.path.join(ravdess_src, folder)

    if os.path.isdir(src_folder):
        ensure(folder)

        for file in os.listdir(src_folder):
            shutil.copy(
                os.path.join(src_folder, file),
                f"{dst}/{folder}/{file}"
            )

print("✅ RAVDESS merged")

# =========================
# 2. FIX CREMA-D
# =========================
for file in os.listdir(cremad_src):
    if file.endswith(".wav"):
        emotion = file.split("_")[2]  # ANG, HAP, etc.

        if emotion in MAP:
            label = MAP[emotion]
            ensure(label)

            shutil.copy(
                os.path.join(cremad_src, file),
                f"{dst}/{label}/{file}"
            )

print("✅ CREMA-D converted")

# =========================
# 3. FIX TESS
# =========================
for folder in os.listdir(tess_src):
    folder_path = os.path.join(tess_src, folder)

    if os.path.isdir(folder_path):

        # extract emotion name
        emotion = folder.split("_")[-1].lower()

        if emotion in MAP:
            label = MAP[emotion]
            ensure(label)

            for file in os.listdir(folder_path):
                shutil.copy(
                    os.path.join(folder_path, file),
                    f"{dst}/{label}/{file}"
                )

print("✅ TESS converted")

print("\n🎉 ALL DATASETS MERGED INTO train_data/")