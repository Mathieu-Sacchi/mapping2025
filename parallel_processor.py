import os
import re
import json
import time
import pandas as pd
import concurrent.futures
from threading import Lock
import google.generativeai as genai
from google.api_core import exceptions as api_exceptions
from dotenv import load_dotenv
import random

# ─── Load environment variables ────────────────────────────────────────────────
load_dotenv()  # Load .env file

# ─── Configuration ────────────────────────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
API_KEYS = os.getenv("GEMINI_API_KEYS", API_KEY).split(',')  # Support multiple API keys
MODEL = os.getenv("MODEL", "gemini-2.0-flash")

EXISTING_FILE = os.getenv("EXISTING_FILE", "existing_mapping.csv")
NEW_FILE = os.getenv("NEW_FILE", "CB.xlsx")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "updated_mapping.csv")
CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", "checkpoint_processed.txt")
ERROR_FILE = os.getenv("ERROR_FILE", "errors_unprocessed.txt")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))  # Number of parallel workers
QUOTA_PER_MINUTE = int(os.getenv("QUOTA_PER_MINUTE", "15"))  # API calls allowed per minute
WAIT_AFTER_QUOTA = int(os.getenv("WAIT_AFTER_QUOTA", "60"))  # Seconds to wait after hitting quota

# Global locks and counters for managing API quotas
api_call_lock = Lock()
checkpoint_lock = Lock()
error_lock = Lock()
output_lock = Lock()
quota_reset_time = time.time() + 60  # Reset quota counter every minute
api_calls_in_window = 0

# Import the prompt template from main.py
from main import PROMPT_TEMPLATE

# ─── Helper: classify with retries ────────────────────────────────────────────
def get_api_key():
    """Get an API key from the pool, rotating if multiple keys are available"""
    return random.choice(API_KEYS)

def safe_classify_entity(name: str, description: str, max_retries: int = MAX_RETRIES) -> tuple[bool, dict]:
    """
    Returns (success_flag, result_dict).
    success_flag = True  → parsed JSON returned
    success_flag = False → nothing parsed; caller must NOT checkpoint this name
    
    This version manages API quotas across multiple workers.
    """
    global api_calls_in_window, quota_reset_time
    
    backoff = 2
    for attempt in range(1, max_retries + 1):
        try:
            # Check quota and wait if needed
            with api_call_lock:
                current_time = time.time()
                
                # Reset counter if we're in a new time window
                if current_time > quota_reset_time:
                    api_calls_in_window = 0
                    quota_reset_time = current_time + 60
                
                # If we've hit our quota for this minute, wait
                if api_calls_in_window >= QUOTA_PER_MINUTE:
                    wait_time = quota_reset_time - current_time
                    print(f"⏸️  Thread quota limit reached, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    # After waiting, recalculate time and reset quota
                    current_time = time.time()
                    api_calls_in_window = 0
                    quota_reset_time = current_time + 60
                
                # Increment the counter and proceed
                api_calls_in_window += 1
            
            # Setup Gemini client with a potentially different API key for each call
            api_key = get_api_key()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL)
                
            prompt = (
                PROMPT_TEMPLATE
                + f"\n\nNow classify the following company:\n"
                + f"Name: {name}\n"
                + f"Description (ID-check only – do NOT rely on it): {description}\n"
            )
            
            resp = model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.05,  # Very low temperature for more consistent formatting and reasoning
                    candidate_count=1,
                    stop_sequences=["</JSON>"],
                    max_output_tokens=2048,
                )
            )
            raw = resp.text.strip()

            # Debug the raw response and search for tags
            print(f"\n🔍 DEBUG - Raw response length for {name}: {len(raw)} chars")
            start_tag = "<JSON>"
            end_tag = "</JSON>"
            
            # Try printing the exact indices to debug
            start_idx = raw.find(start_tag)
            end_idx = raw.find(end_tag)
            print(f"🔍 DEBUG - Tag indices: start_tag={start_idx}, end_tag={end_idx}")
            
            # If standard search fails, try a more forceful approach
            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                print(f"\n🔍 DEBUG - Standard tag search failed, trying direct JSON extraction")
                # Try to find any JSON-like content
                json_pattern = r'({.*?})'
                json_matches = re.findall(json_pattern, raw, re.DOTALL)
                
                if json_matches:
                    print(f"🔍 DEBUG - Found {len(json_matches)} potential JSON objects")
                    # Try the first match that looks promising
                    for potential_json in json_matches:
                        try:
                            # Clean up and try to parse
                            clean_json = potential_json.replace("{{", "{").replace("}}", "}")
                            clean_json = re.sub(r',\s*}', '}', clean_json)
                            clean_json = clean_json.replace("'", '"')
                            result = json.loads(clean_json)
                            print(f"🔍 DEBUG - Successfully parsed JSON from pattern match!")
                            return True, result
                        except json.JSONDecodeError:
                            continue
                
                # If we still can't find valid JSON, print the raw response for manual inspection
                print(f"\n🔍 DEBUG - Raw response for {name}:\n{raw}\n")
                raise ValueError("No valid JSON block found")
            
            # Extract the JSON text
            json_text = raw[start_idx + len(start_tag):end_idx].strip()
            print(f"🔍 DEBUG - Extracted JSON text length: {len(json_text)} chars")
            
            # Handle double braces
            if json_text.startswith("{{") and json_text.endswith("}}"):
                json_text = json_text[1:-1]
                print("🔍 DEBUG - Removed double braces")
            
            # Clean up common JSON formatting issues
            original_json = json_text
            json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
            json_text = json_text.replace("'", '"')  # Replace single quotes with double quotes
            
            if original_json != json_text:
                print("🔍 DEBUG - Cleaned up JSON formatting")
            
            # Try parsing the JSON with multiple fallback approaches
            try:
                result = json.loads(json_text)
                print("🔍 DEBUG - Successfully parsed JSON!")
                return True, result
            except json.JSONDecodeError as e:
                print(f"\n🔍 DEBUG - Error parsing JSON: {e}")
                print(f"🔍 DEBUG - Extracted JSON for {name}:\n{json_text}\n")
                
                # Fallback: Try forcing the structure
                try:
                    print("🔍 DEBUG - Trying forceful JSON parsing...")
                    # Attempt to force the JSON into a valid structure
                    cleaned_json = re.sub(r'[^\x00-\x7F]+', '', json_text)  # Remove non-ASCII chars
                    cleaned_json = re.sub(r'[\n\r\t]+', ' ', cleaned_json)  # Normalize whitespace
                    result = json.loads(cleaned_json)
                    print("🔍 DEBUG - Forceful JSON parsing succeeded!")
                    return True, result
                except json.JSONDecodeError:
                    # Last attempt: Try to construct a minimal valid JSON
                    print("🔍 DEBUG - Trying minimal JSON reconstruction...")
                    minimal_json = {
                        "is_startup": False,
                        "is_startup_confidence": 0,
                        "startup_rationale": "Parsing error",
                        "is_gen_ai_startup": False,
                        "is_gen_ai_startup_confidence": 0,
                        "gen_ai_rationale": "Parsing error",
                        "layer": None,
                        "layer_confidence": 0,
                        "category": None,
                        "category_confidence": 0,
                        "is_linked_to_france": False,
                        "is_linked_to_france_confidence": 0
                    }
                    
                    # Try to extract values from the text using regex
                    try:
                        confidence_matches = re.findall(r'"([^"]+)":\s*(\d+)', json_text)
                        bool_matches = re.findall(r'"([^"]+)":\s*(true|false)', json_text.lower())
                        string_matches = re.findall(r'"([^"]+)":\s*"([^"]+)"', json_text)
                        
                        # Update the minimal JSON with any values we could extract
                        for key, value in confidence_matches:
                            if key in minimal_json:
                                minimal_json[key] = int(value)
                        
                        for key, value in bool_matches:
                            if key in minimal_json:
                                minimal_json[key] = value == 'true'
                        
                        for key, value in string_matches:
                            if key in minimal_json:
                                minimal_json[key] = value
                                
                        print("🔍 DEBUG - Created minimal JSON with extracted values")
                        return True, minimal_json
                    except Exception as e:
                        print(f"🔍 DEBUG - Minimal JSON reconstruction failed: {e}")
                        raise ValueError(f"Failed to parse JSON: {e}")
                raise
        except api_exceptions.ResourceExhausted as e:
            # Handle rate limit
            retry_delay = WAIT_AFTER_QUOTA  # Default to configured value
            if hasattr(e, 'retry_delay') and e.retry_delay:
                retry_delay = e.retry_delay.seconds
            print(f"⏸️  Rate limit hit, waiting {retry_delay}s...")
            time.sleep(retry_delay)
            continue
        except api_exceptions.ServiceUnavailable as e:
            code = getattr(e, "code", 503)
            if attempt < max_retries and code in (500, 502, 503, 504):
                wait = backoff * attempt
                print(f"[Retry {attempt}/{max_retries}] {code} – wait {wait}s")
                time.sleep(wait)
                continue
            print(f"❌  ServiceUnavailable {code} for {name}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌  Parse error for {name}: {e}")
            if attempt < max_retries:
                wait = backoff * attempt
                print(f"[Retry {attempt}/{max_retries}] Parse error – wait {wait}s")
                time.sleep(wait)
                continue
        except api_exceptions.BadRequest as e:
            print(f"❌  BadRequest for {name}: {e}")
        except Exception as e:
            print(f"❌  Unexpected error for {name}: {e}")
        break                            # → exit retry loop

    # after all retries failed
    return False, {}                    # caller will log + retry later

def process_company(row):
    """Process a single company - used by worker threads"""
    name = row["Company Name"]
    desc = row["Description"]
    
    # Skip if already processed in a previous run
    with checkpoint_lock:
        if name in processed_set:
            return None
    
    print(f"Processing: {name}")
    success, result = safe_classify_entity(name, desc)
    
    if not success:
        # Keep a log of items that still need processing; do NOT checkpoint the name
        with error_lock:
            with open(ERROR_FILE, "a", encoding="utf-8") as err:
                err.write(name + "\n")
        return None
    
    # ► Successful parse → checkpoint immediately
    with checkpoint_lock:
        with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
            f.write(name + "\n")
        processed_set.add(name)
    
    # Collect Gen-AI startups for the final mapping
    if result.get("is_startup") and result.get("is_gen_ai_startup"):
        record = {
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
        }
        return record
    
    return None

def main():
    global processed_set
    
    # ─── Load datasets ───────────────────────────────────────────────────────────
    existing_df = pd.read_csv(EXISTING_FILE)
    new_df = pd.read_excel(NEW_FILE)        # Name + Description
    
    # keep first two columns only
    new_df = new_df.iloc[:, :2]
    new_df.columns = ["Company Name", "Description"]
    
    todo_df = new_df[~new_df["Company Name"].isin(existing_df["Company Name"])]
    print(f"Found {len(todo_df)} new companies to process")
    
    # ─── Checkpoint ───────────────────────────────────────────────────────────────
    processed_set = set()
    if os.path.exists(CHECKPOINT_FILE):
        processed_set = {line.strip() for line in open(CHECKPOINT_FILE, encoding="utf-8")}
    
    # ─── Parallel processing ──────────────────────────────────────────────────────
    records = []
    
    # Convert DataFrame to list of dictionaries for parallel processing
    companies_to_process = todo_df.to_dict('records')
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_company = {executor.submit(process_company, company): company for company in companies_to_process}
        
        for future in concurrent.futures.as_completed(future_to_company):
            company = future_to_company[future]
            try:
                record = future.result()
                if record:
                    with output_lock:
                        records.append(record)
                        # Periodically save intermediate results
                        if len(records) % 10 == 0:
                            intermediate_df = pd.concat([existing_df, pd.DataFrame(records)], ignore_index=True)
                            intermediate_df.to_csv(OUTPUT_FILE, index=False)
                            print(f"✅  Intermediate save: {len(records)} Gen-AI startups → {OUTPUT_FILE}")
            except Exception as e:
                print(f"❌  Error processing {company['Company Name']}: {e}")
    
    # ─── Save output ─────────────────────────────────────────────────────────────
    updated_df = (
        pd.concat([existing_df, pd.DataFrame(records)], ignore_index=True)
        if records else existing_df
    )
    updated_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅  Added {len(records)} Gen-AI startups → {OUTPUT_FILE}")
    print(f"🔖  Progress checkpoint saved to {CHECKPOINT_FILE}")

if __name__ == "__main__":
    main() 