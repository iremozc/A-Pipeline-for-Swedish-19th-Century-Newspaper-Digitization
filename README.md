# A-Pipeline-for-Swedish-19th-Century-Newspaper-Digitization
# Historical Newspaper Digitization Pipeline

This project implements a modular pipeline to process 19th-century Swedish newspaper scans — turning raw images into structured, searchable articles. It was developed as part of my Master's thesis at Uppsala University.

---

##  What This Project Does

Given a scanned newspaper page, this pipeline:

1. **Detects layout blocks** (like paragraphs and non-text) using YOLOv8
2. **Segments text lines**
3. **Runs Optical Character Recognition (OCR)** using a fine-tuned TrOCR model
4. **Groups lines into full articles** using semantic similarity and clustering

It is built using Python, Hugging Face Transformers, and YOLO.



###  Main Components

#### `article_segmentation/`
Scripts for segmenting and clustering OCR text into articles using modern and historical word embeddings, BERT, and clustering algorithms.
- `historical_embeddings_article_seg.py`: Article segmentation using historical embeddings.
- `concat_article_seg.py`: Article segmentation with concatenated embeddings.
- `bert_article_segmentation.py`: Article segmentation using BERT-based models.

#### `trocr-finetune/`
Scripts for fine-tuning the TrOCR model (Hugging Face Transformers) on Swedish Fraktur handwriting datasets.
- `tr-ocr-fine-tune.py`: Fine-tuning pipeline for TrOCR.


#### `yolov8-finetune/`
Scripts and weights for fine-tuning YOLOv8 for layout block detection in historical newspapers.
- `yolov8-fine-tuning-using-optuna.py`: Fine-tuning and hyperparameter optimization with Optuna.

---

---

## Citation
If you use this pipeline, please cite the corresponding thesis or repository.

---
## 🔗 Model Downloads

- **YOLOv8 Fine-tuned Model:**
  - Download or use directly from the Hugging Face Hub: [Iremozcelik/yolov8x_fine_tuned_swedish_19thcentury](https://huggingface.co/Iremozcelik/yolov8x_fine_tuned_swedish_19thcentury)
- **TrOCR Fine-tuned Model:**
  - Download or use directly from the Hugging Face Hub: [Iremozcelik/trOCR_fine_tuned_SwedishFraktur](https://huggingface.co/Iremozcelik/trOCR_fine_tuned_SwedishFraktur)
