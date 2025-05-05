# French Gen-AI Startup Mapping

This project identifies and classifies French startups leveraging Generative AI technologies. The system uses the Gemini API to analyze company data and determine whether each company is a startup using Gen-AI technology, along with additional metadata like layer, category, and French connection.

## Features

- Automated classification of companies as startups and Gen-AI startups
- Detailed categorization by technology layer and business category
- Confidence scoring for each classification
- Checkpoint system to resume processing after interruptions
- Error handling and retry mechanism
- Rate limit management

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/Mathieu-Sacchi/mapping2025.git
   cd mapping2025
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on the example:
   ```
   cp .env.example .env
   ```

4. Edit `.env` and add your Gemini API key.

## Usage

1. Prepare your data:
   - Place your existing mappings in `existing_mapping.csv`
   - Place new companies to analyze in `CB.xlsx`

2. Run the main script:
   ```
   python main.py
   ```

3. For reprocessing errors:
   ```
   python reprocess_errors.py
   ```

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key
- `MODEL`: Gemini model to use (default: gemini-2.0-flash)
- `EXISTING_FILE`: Path to existing companies CSV
- `NEW_FILE`: Path to new companies Excel file
- `OUTPUT_FILE`: Path to output CSV file
- `CHECKPOINT_FILE`: Path to checkpoint file
- `ERROR_FILE`: Path to error log file
- `MAX_RETRIES`: Maximum API retry attempts
- `BATCH_SIZE`: Number of companies to process before pausing
- `PAUSE_SECONDS`: Seconds to pause between batches

## File Structure

- `main.py`: Main processing script
- `reprocess_errors.py`: Script for reprocessing failed items
- `requirements.txt`: Python dependencies
- `.env.example`: Template for environment variables
- `.gitignore`: Git ignore rules
- `existing_mapping.csv`: Existing classified companies
- `updated_mapping.csv`: Output file with newly classified companies

## Error Handling

The system handles several types of errors:
- API rate limiting
- Transient server errors
- Parse errors in responses
- Bad requests

Failed items are logged to `errors_unprocessed.txt` for later reprocessing.

## License

MIT 