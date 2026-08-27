# NeuroScan 

Brain tumour classifier trained on MRI scans. Upload a scan, get a prediction across 4 classes: glioma, meningioma, pituitary tumor, or no tumor.

**[Try it live →](https://neurroscan.streamlit.app)**

> Educational project only. Not a medical device. Don't use this to make health decisions.

## Results

| Model | Accuracy | Loss |
|---|---|---|
| CNN (scratch) | 85.94% | 1.22 |
| **VGG16 (fine-tuned)** | **90.44%** | **0.61** |

Transfer learning won on both counts. The loss gap is the interesting part: it's roughly half, so the fine-tuned model isn't just more accurate, it's more honest about its confidence. The scratch CNN was often confidently wrong.

<details>
<summary><b>Per-class breakdown (baseline CNN)</b></summary>

| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma | 0.96 | 0.72 | 0.82 |
| meningioma | 0.82 | 0.80 | 0.81 |
| notumor | 0.76 | 1.00 | 0.87 |
| pituitary | 0.95 | 0.92 | 0.93 |

Glioma recall of 0.72 is the weak spot. The model missed a quarter of gliomas and most of those got called "notumor," which is the worst kind of miss to make here.
</details>

## How it works

Two models, trained on the same data, so transfer learning could be measured instead of assumed:

1. **Baseline CNN**: 3 conv blocks (32→64→128 filters), trained from scratch with augmentation.
2. **VGG16**: pretrained on ImageNet, frozen for 20 epochs, then fine-tuned (last 4 layers unfrozen, learning rate dropped to `1e-5`). Stopped early at epoch 7/15, 95.89% val accuracy.

That low fine-tuning learning rate matters. Crank it up and the random weights in the new head send gradients that wreck the pretrained filters you were trying to keep.

<details>
<summary><b>Preprocessing (read this before you touch the code)</b></summary>

VGG16 uses `preprocess_input`, not `Rescaling(1./255)`. It converts RGB to BGR and subtracts ImageNet channel means, so values land around `[-124, 151]`. If you divide by 255 first, the model won't crash, it'll just quietly get worse. The baseline CNN uses the opposite pipeline (rescaling, no `preprocess_input`). Don't mix them up.
</details>

<details>
<summary><b>Architecture</b></summary>

```
VGG16 (frozen base, last 4 layers unfrozen)
→ Flatten
→ Dense(128, relu)
→ Dropout(0.5)
→ Dense(4, softmax)
```

17.9M parameters, 10.3M trainable during fine-tuning.

**Why the model file is 68 MiB, not 147 MiB:** Keras saves Adam's optimizer state (78.5 MiB) by default, and inference never needs it. `slim_model.py` strips it out, verifying all 32 weight arrays stay bit-identical first. That's what keeps the file under GitHub's 100 MB limit with no Git LFS.
</details>

## Dataset

| Split | Per class | Total |
|---|---|---|
| Training | 1,400 | 5,600 |
| Testing | 400 | 1,600 |

Not committed here (adds ~175 MB). Grab it from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and arrange as `Training/<class>/` and `Testing/<class>/`. Four sample scans are in `samples/` if you just want to try the app.

## Run it

```bash
git clone https://github.com/Alok-0601/NeuroScan.git
cd NeuroScan
pip install -r requirements.txt
streamlit run app.py
```

Upload a scan from `samples/`, or your own JPG/PNG.

<details>
<summary><b>Deployment gotchas (if you're forking this)</b></summary>

- **`requirements.txt` must be at repo root**, or you get a `ModuleNotFoundError` that looks like a code bug but isn't.
- **Don't use Git LFS for the model.** Streamlit Cloud clones without resolving LFS, so you'd get a 134-byte pointer file instead of the model.
- **Memory is tight on free tier.** `app.py` loads with `compile=False` and caches with `st.cache_resource` to stay under budget.
- **Pin the theme** in `.streamlit/config.toml`, or the app's light-mode CSS breaks under a visitor's dark mode.
- Use `tensorflow-cpu`, not `tensorflow`. The GPU build drags in ~2GB of CUDA libraries you'll never use.
</details>

## What it doesn't tell you

* 90.44% accuracy means 1 in 10 scans gets misclassified, and the fine-tuned model's per-class errors haven't been broken down yet.
* Some meningioma images are augmented (100/1,400 train, 103/400 test), so part of that class's score rests on synthetic variants. No duplicates between train/test though, confirmed by hashing.
* One 2D slice, no clinical context, one dataset. Performance on scans from other scanners or populations is unknown and probably worse.

## License

MIT for the code and model. The MRI dataset has its own license, see the [Kaggle page](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
