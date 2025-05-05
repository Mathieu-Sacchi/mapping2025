# Gen-AI Startup Mapping Project Analysis

## Project Overview
The project aims to maintain a comprehensive database of French startups leveraging Generative AI technologies, distinguishing them from traditional AI startups. It uses Google's Gemini API to classify companies based on structured criteria:

1. Startup Status (is_startup)
2. Gen-AI Status (is_gen_ai_startup)
3. Layer (Foundational/Infrastructure/Application)
4. Category (specific industry categories)
5. France Link (HQ, founders, team in France)

## Current Implementation Analysis

### Core Workflow
- Python script (`main.py`) loads company data from CSV/Excel files
- Classifies companies using Gemini 2.0-flash API with Google Search integration
- Implements error handling and exponential backoff for API retries
- Tracks processed companies via checkpoints
- Logs errors for manual review
- Outputs results to an updated mapping file

### Strengths
- Robust error handling and retry mechanism in `safe_classify_entity`
- Checkpoint system to prevent reprocessing previously examined companies
- Rate limiting implementation (15 successful calls, then 60s pause)
- Clear prompt design with concrete examples for the Gemini model
- JSON validation to ensure data integrity

### Areas for Improvement

#### 1. Error Handling and Reprocessing
- No dedicated mechanism to reprocess companies in `errors_unprocessed.txt`
- Duplicates in the error file due to append-only operation
- No error categorization (API errors vs parsing errors)

#### 2. Data Handling
- Different scripts have slightly different approaches to column handling
- Inconsistent handling of input formats (semicolon separation parsing differs)
- No data validation or cleaning before processing

#### 3. Efficiency and Optimization
- No tracking of API usage or costs
- Limited parallelization could be improved for throughput
- No caching mechanism for previous Gemini responses

#### 4. Code Structure
- Some code duplication between `main.py` and `Code Used for Attio.py`
- Limited modularization and reusable components
- Hardcoded values throughout (API key, file paths, etc.)

## Recommendations

### 1. Create Error Reprocessing Functionality
```python
def reprocess_errors(error_file="errors_unprocessed.txt", max_per_run=50):
    """Process previously failed items from the error file."""
    if not os.path.exists(error_file):
        print(f"Error file {error_file} not found.")
        return
    
    # Read and deduplicate error file
    with open(error_file, "r", encoding="utf-8") as f:
        failed_names = list(dict.fromkeys(line.strip() for line in f if line.strip()))
    
    print(f"Found {len(failed_names)} unique failed items to reprocess.")
    
    # Limit number of reprocessed items per run
    to_process = failed_names[:max_per_run]
    successful = []
    still_failed = []
    
    # Get company details from source files
    combined_df = pd.concat([
        pd.read_excel(NEW_FILE),
        pd.read_csv(EXISTING_FILE)
    ], ignore_index=True)
    
    # Create a map for faster lookups
    company_map = {
        name: desc for name, desc in 
        zip(combined_df["Company Name"], combined_df["Description"])
    }
    
    records = []
    batch_counter = 0
    
    for name in to_process:
        desc = company_map.get(name, "")
        
        if not desc:
            print(f"Warning: No description found for {name}")
        
        success, result = safe_classify_entity(name, desc)
        
        if success:
            successful.append(name)
            # Add to results if it's a Gen-AI startup
            if result.get("is_startup") and result.get("is_gen_ai_startup"):
                records.append({
                    "Company Name": name,
                    "Description": desc,
                    "Layer": result.get("layer"),
                    "Layer Confidence": result.get("layer_confidence"),
                    "Category": result.get("category"),
                    "Category Confidence": result.get("category_confidence"),
                    "Startup Confidence": result.get("is_startup_confidence"),
                    "GenAI Confidence": result.get("is_gen_ai_startup_confidence"),
                    "Linked to France": result.get("is_linked_to_france"),
                    "France Confidence": result.get("is_linked_to_france_confidence"),
                })
            
            # Checkpoint successful reprocessing
            with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
                f.write(name + "\n")
        else:
            still_failed.append(name)
        
        # Rate limiting
        batch_counter += 1
        if batch_counter >= 15:
            print("⏸️  Pausing 60s to respect rate limits...")
            time.sleep(60)
            batch_counter = 0
    
    # Update output file with newly processed items
    if records:
        existing_df = pd.read_csv(OUTPUT_FILE)
        updated_df = pd.concat([existing_df, pd.DataFrame(records)], ignore_index=True)
        updated_df.to_csv(OUTPUT_FILE, index=False)
    
    # Rewrite error file with remaining failures
    remaining_failures = list(set(failed_names) - set(successful))
    with open(error_file, "w", encoding="utf-8") as f:
        for name in remaining_failures:
            f.write(name + "\n")
    
    print(f"Reprocessing complete: {len(successful)} succeeded, {len(still_failed)} still failed")
    print(f"Added {len(records)} new Gen-AI startups to the mapping")
    return records
```

### 2. Improve Project Structure

```
mappingAI25/
├── config/
│   ├── settings.py         # API keys, file paths, configurations
│   └── categories.py       # Category definitions and validation
├── data/
│   ├── input/              # Input files
│   ├── output/             # Output files
│   └── logs/               # Error logs and processing logs
├── src/
│   ├── __init__.py
│   ├── api.py              # Gemini API interactions
│   ├── processor.py        # Core processing logic
│   ├── data_handler.py     # Data loading, cleaning, validation
│   └── error_handler.py    # Error processing and reprocessing
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py       # Checkpoint management
│   └── validation.py       # Data validation utilities
├── scripts/
│   ├── process_new.py      # Process new companies
│   ├── reprocess_errors.py # Reprocess failed items
│   └── generate_report.py  # Generate statistics and reports
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

### 3. Enhance Data Validation and Processing

- Implement input validation for company data
- Add logging with different severity levels
- Create detailed reports on processing results
- Implement better duplicate detection in source data
- Add data enrichment from additional sources

### 4. Optimize API Usage

- Add caching of Gemini responses to avoid reprocessing identical queries
- Implement more sophisticated rate limiting based on actual API response times
- Add detailed tracking of API costs and usage
- Consider implementing async processing for better throughput

### 5. Improve Error Handling

- Categorize errors by type (API, parsing, validation)
- Add enhanced logging with timestamps and stack traces
- Create a web dashboard for manual review of uncertain classifications
- Implement periodic batch reprocessing of error file

## Implementation Priority

1. Error reprocessing functionality (highest priority)
2. Code refactoring for better modularization
3. Data validation and enrichment
4. API usage optimization
5. Enhanced error categorization

These improvements will make the system more robust, easier to maintain, and more efficient at handling the classification of Gen-AI startups. 