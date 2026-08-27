# NeuroScan — Brain Tumor Detection from MRI

**[Live app →](https://neurroscan.streamlit.app)**

A four-class brain tumour classifier built on VGG16 transfer learning, with a Streamlit interface for single-scan inference.

The model sorts a brain MRI slice into **glioma**, **meningioma**, **pituitary tumour**, or **no tumour**, and reports the probability it assigned to every class rather than only the winning one.

> **This is an educational project, not a medical device.** It has not been clinically validated and must not be used to make decisions about anyone's health.

---

## Results

Two models were trained on the same data so the benefit of transfer learning could be measured rather than assumed. Both were evaluated on the held-out 1,600-image test set.

| Model | Test accuracy | Test loss | Precision (w) | Recall (w) | F1 (w) |
|---|---|---|---|---|---|
| CNN from scratch | 85.94% | 1.2157 | 0.8733 | 0.8594 | 0.8582 |
| **VGG16 fine-tuned** | **90.44%** | **0.6134** | **0.9148** | **0.9044** | **0.9025** |

Transfer learning bought about 4.5 points of accuracy, but the more telling number is loss: it fell by roughly half, meaning the fine-tuned model is not just more often right but substantially better calibrated. The scratch CNN was overconfident when it was wrong.

The per-class breakdown for the baseline CNN shows where it struggled, and why accuracy alone would have been misleading:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma | 0.96 | 0.72 | 0.82 |
| meningioma | 0.82 | 0.80 | 0.81 |
| notumor | 0.76 | 1.00 | 0.87 |
| pituitary | 0.95 | 0.92 | 0.93 |

Glioma recall of 0.72 is the problem case: the baseline missed more than a quarter of gliomas, and those misses largely landed in `notumor`, which is the most costly direction for an error to go in this domain. Precision of 0.96 alongside it means the model was cautious about calling glioma — when it did, it was usually right, but it stayed quiet too often.

## Approach

**Baseline.** A three-block convolutional network trained from scratch (32 → 64 → 128 filters, each followed by max pooling, then a 128-unit dense layer with 0.5 dropout). Light augmentation — random rotation, zoom, and translation — was applied inside this model, along with `Rescaling(1./255)`.

**Transfer learning.** VGG16 pre-trained on ImageNet with `include_top=False`, followed by flatten, a 128-unit ReLU layer, 0.5 dropout, and a 4-way softmax. Training ran in two stages:

1. **Frozen base**, 20 epochs, Adam at default learning rate. The convolutional stack acts purely as a feature extractor while the new head learns.
2. **Fine-tuning**, with the last four VGG16 layers unfrozen and Adam dropped to `1e-5`. Early stopping halted this at epoch 7 of 15, reaching 95.89% validation accuracy.

The low fine-tuning learning rate is the important detail — at the default rate the large gradients from a randomly initialised head would wash out the pretrained filters that make transfer learning worth doing in the first place.

Note that augmentation was deliberately *not* carried into the VGG16 model; it relied on pretrained features plus dropout for regularisation instead.

## Preprocessing

The VGG16 model uses **only** `preprocess_input` from `tensorflow.keras.applications.vgg16`. There is no `Rescaling(1./255)` layer anywhere in it.

```
uploaded MRI → resize 224×224 → img_to_array → expand_dims → preprocess_input → model
```

This matters more than it looks. VGG16's `preprocess_input` uses Caffe-style normalisation: it converts RGB to BGR and subtracts the ImageNet channel means, producing values in roughly `[-124, 151]`. Dividing by 255 first would hand the network inputs near zero and predictions would silently degrade rather than fail loudly. The baseline CNN *does* use a rescaling layer, so the two pipelines are not interchangeable.

## Model architecture

```
Sequential
├── VGG16  (ImageNet weights, include_top=False, 224×224×3)
│     └── last 4 layers unfrozen during fine-tuning
├── Flatten
├── Dense(128, relu)
├── Dropout(0.5)
└── Dense(4, softmax)
```

Saved with Keras **3.13.2** as `Models/brain_tumor_vgg16.keras` (68 MiB). Loading it requires Keras 3.13 or newer.

The model has 17,926,596 parameters, of which 10,291,332 were trainable during fine-tuning. That second number is worth noting, because Keras saves the optimiser's state alongside the weights by default and Adam keeps two buffers per trainable parameter:

| Stored in the `.keras` archive | Values | Size |
|---|---|---|
| Model weights | 17,926,596 | 68.4 MiB |
| Adam momentum + variance buffers | 20,582,664 | 78.5 MiB |
| **Original file** | | **146.9 MiB** |

Over half the original file was training bookkeeping that inference never reads — it exists so a training run can be resumed. `slim_model.py` strips it by loading with `compile=False` and re-saving, which leaves every weight untouched (the script verifies all 32 weight arrays are bit-identical before it replaces anything). That brings the file under GitHub's 100 MB limit, so it is committed as an ordinary file with no Git LFS involved.

## Dataset

Four balanced classes of brain MRI slices, split into a training and a testing directory:

| Split | glioma | meningioma | notumor | pituitary | Total |
|---|---|---|---|---|---|
| Training | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| Testing | 400 | 400 | 400 | 400 | 1,600 |

Training was further divided 80/20 into 4,480 training and 1,120 validation images. Source images vary in resolution (192×192 up to 680×680) and are a mix of grayscale and RGB, so everything is resized to 224×224 and read as three-channel at load time.

The image folders are **not** committed to this repository — they are a redistribution of a public dataset and would add ~175 MB and 7,200 files to the history. Download the four-class brain tumour MRI dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and arrange it as `Training/<class>/` and `Testing/<class>/`. Note that the split used here was rebalanced to exactly 1,400/400 per class, so counts will differ slightly from the upstream version.

Four sample scans, one per class, are included in `samples/` so the app can be tried immediately without downloading anything.

## Project structure

```
NeuroScan/
├── app.py                          # Streamlit application
├── slim_model.py                   # one-off: strips optimiser state from the model
├── requirements.txt                # dependencies — Streamlit Cloud reads this file
├── Brain.ipynb                     # training, evaluation, and model comparison
├── .streamlit/
│   └── config.toml                 # pinned light theme, upload size limit
├── Models/
│   ├── brain_tumor_vgg16.keras     # fine-tuned model (68 MiB, plain file — no LFS)
│   └── class_names.json            # class order — the app reads this, never hardcodes it
├── samples/                        # one demo MRI per class
├── Training/                       # not committed — see Dataset
└── Testing/                        # not committed — see Dataset
```

## Getting started

```bash
git clone https://github.com/Alok-0601/NeuroScan.git
cd NeuroScan
pip install -r requirements.txt
streamlit run app.py
```

Then upload any scan from `samples/`, or your own JPG/JPEG/PNG.

`requirements.txt` asks for `tensorflow-cpu` rather than `tensorflow` on purpose — the full package pulls in roughly 2 GB of CUDA libraries that CPU-only inference will never touch, and on a hosted instance that download alone can exhaust the build limits. The `keras>=3.13.2` floor matters because the `.keras` archive was written by that version and older ones may refuse to deserialise it.

## Deployment notes

The app is a single Streamlit script with no server-side state, so it deploys to Streamlit Community Cloud by pointing at `app.py` on the `main` branch. Four things are easy to get wrong, all of which this repository is arranged to avoid:

**`requirements.txt` must be at the repository root.** Streamlit Cloud installs nothing without it, and the failure surfaces as `ModuleNotFoundError: No module named 'tensorflow'` on the import line — which reads like a bug in the code rather than a missing dependency.

**The model must not be in Git LFS.** Streamlit Cloud clones without resolving LFS objects, so an LFS-tracked model arrives as a 134-byte pointer file. This project hit exactly that, which is why `slim_model.py` exists: a 68 MiB model needs no LFS and commits as an ordinary binary. `app.py` also checks the file size at load time and reports a pointer file explicitly, because the alternative is an unreadable deserialisation error. `.gitattributes` documents why no LFS filter should be re-added.

**Memory is the binding constraint.** TensorFlow's runtime plus VGG16's weights is a meaningful fraction of a free-tier instance's RAM. `app.py` loads with `compile=False` so Adam's 78.5 MiB of state is never allocated, and caches the model with `st.cache_resource` so it is read once per session rather than once per interaction.

**The theme needs pinning.** `app.py` ships a stylesheet written against a light background. Without `.streamlit/config.toml` the app inherits the visitor's Streamlit theme, and under a dark default the built-in widget text and file uploader become nearly unreadable.

## The application

`app.py` loads the trained model once and caches it, reads the class order from `class_names.json` rather than hardcoding it, and runs the preprocessing chain described above. The result panel shows the predicted class and model confidence, then all four probabilities in a fixed order with a stable colour per class.

Two design decisions worth calling out. The interface reports a **margin over next** figure alongside confidence, because 94% against 92% and 94% against 2% are very different outcomes and confidence alone hides which one occurred. And below 60% confidence it explicitly says the prediction is not decisive instead of presenting a near-tie as settled — the notebook produced a 51.9% prediction on a real test image, and displaying that as an answer would misrepresent what the model actually did.

The word used throughout is **model confidence**, never diagnostic certainty. It is the probability the softmax layer assigned to one class, nothing more.

## Limitations

Test accuracy of 90.44% means roughly one scan in ten is misclassified, and the errors are not evenly distributed — the baseline's weakest class was glioma recall, and the fine-tuned model was not broken down per class, so its own failure modes are less well characterised. Anyone extending this should generate that breakdown first.

The training data contains a small number of pre-augmented images: 100 of the 1,400 meningioma training images and 103 of the 400 meningioma test images have `aug` in their filenames, meaning the meningioma test score rests partly on synthetic variants rather than distinct scans. No image appears identically in both splits — verified by content hashing, which found zero exact duplicates across Training and Testing — but the augmented subset is worth knowing about before quoting the meningioma numbers.

More fundamentally, the model sees a single 2D slice with no clinical context, no patient history, and no sequence information, and it was trained on one curated dataset. Performance on scans from different scanners, protocols, or populations is unknown and should be assumed worse.

## License

Released under the MIT License. See [LICENSE](LICENSE).

The MRI images are from a third-party public dataset and are subject to their original terms; the license here covers the code and the trained model only.
