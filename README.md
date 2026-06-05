# Missing Persons - Armed Conflict Telegram Pipeline
A research pipeline for extracting, detecting and structuring missing persons reports from Arabic-language 
Telegram channels. The system combines NLP and Computer Vision to extract, analyse and organize missing case data 
into a PostgreSQL database, accessible via an interactive Dash dashboard.


## Project Structure
```
├── data/               # Raw and cleaned Telegram data, annotated gold datasets (not uploaded in git)
├── db/                 # Database schema and setup scripts, incl. script for Dash dashboard application
├── outputs/            # Model outputs and exports (not uploaded in git)
├── model_output/       # Trained AraBERT model (not uploaded in git)
├── scripts/            # Data preprocessing and pipeline execution scripts for CV and NLP, model training execution scripts (incl. evaluation)
└── src/                # Pipeline source code for each pipeline component: preprocessing, NLP, CV, and evaluation 
```


## Running the pipeline 
### Note
Create a .env file in the project root with the following variables:
```
GROQ_API_KEY=your_value
DB_URI=your_value
API_ID=your_value (Telegram)
API_HASH=your_value (Telegram)
```

### 1 Install dependencies:
```bash
pip install -r requirements.txt
```

### 2 NLP Pipeline (configure dataset, timeframe, etc. in pipeline scripts)
- Preprocessing to extract messages via Telegram API and preprocess Arabic language
- Sequence Classification to detect whether a message relates to a missing person
- Named Entity Recognition (NER): Extracts name of missing person, location, date, and age
- Translation: Arabic to English
- Clustering: Groups messages that refer to the same missing person
- Pseudonymization: Replaces sensitive identifiers with pseudonyms in Arabic clean_text

**To run the NLP pipeline:**
```bash
python scripts/run_nlp_preprocessing.py
python scripts/run_nlp_pipeline.py
python scripts/run_nlp_evaluation.py
```

### 3 Computer Vision Pipeline (Experimental only)
- Face detection: Detects if face in image or not 
- Color detection: Extract dominant color features of shirt region
- Image similarity: Compares images across the folder to find potential matches 

**To run the CV pipeline:**
```bash
python scripts/run_cv_preprocessing.py
python scripts/run_cv_pipeline.py
python scripts/run_cv_evaluation.py
```


### 4 Database (PostgreSQL) and App Dashboard (Dash) Setup
**Prerequisites:** PostgreSQL installed and running locally.

**Create the database**
```sql
CREATE DATABASE missing_persons;
```

**Run schema setup**
```bash
psql -U postgres -d missing_persons -f db/schema.sql
```

**Populate the database:**
```bash
python db/run_postgre_development.py
```

**Run the App dashboard with Dash:**
```bash
python app.py
```
Then open [http://localhost:8050](http://localhost:8050) in your browser.
The dashboard is password protected. Password available upon request to the researcher.
For access to data and model outputs, contact: madita.schulte@iu-study.org
