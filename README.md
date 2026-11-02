# AAI-003 Customer Review Sentiment Analysis

**Institution:** Invertis University, Bareilly  
**Programme:** BCA AI | Semester V | Section A  
**Mentor:** Sachin Kumar  
**Team Leader:** Virendra Pal

## Project overview
A supervised NLP project that classifies English customer reviews as Positive or Negative. It compares TF-IDF + Logistic Regression with TF-IDF + Multinomial Naive Bayes.

## Dataset
UCI Sentiment Labelled Sentences (Amazon, IMDb, Yelp):  
https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences

The original dataset provides binary labels (1 = Positive, 0 = Negative), not a Neutral class. The notebook downloads the dataset when run. It uses review text only as the model feature. Customer ratings and identifiers are not used.

## Folder structure
- `notebook/` — complete end-to-end notebook
- `data/` — dataset extraction folder created during notebook execution
- `outputs/` — cleaned data, metrics, charts, predictions, error review sheet
- `model/` — saved model pipeline
- `app.py` — optional Streamlit prediction interface
- `requirements.txt` — Python dependencies
- `TEAM_TASK_ALLOCATION.xlsx` — member responsibilities
- `Project_Report.docx` — report template with honest metric placeholders
- `Project_Presentation.pptx` — presentation template with metric placeholders

## Run in Google Colab
1. Upload `notebook/AAI003_Customer_Review_Sentiment_Analysis.ipynb`.
2. Select Runtime > Run all.
3. Allow the UCI dataset download.
4. Check that every cell completes without error.
5. Review at least five incorrect predictions and complete `outputs/error_review_to_complete.csv`.
6. Copy actual metrics from `outputs/test_model_comparison.csv` into the report and slides.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Optional prediction app
After running the notebook and generating `model/sentiment_model_pipeline.joblib`:
```bash
streamlit run app.py
```

## Important academic note
No project can guarantee a particular grade. The included report and slides intentionally leave final performance values blank until the notebook is freshly executed. Do not claim 100% accuracy or invent results. Explain binary scope, dataset limitations, error examples, and human review.
