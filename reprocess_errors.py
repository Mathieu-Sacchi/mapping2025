import os, re, json, time
import pandas as pd
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.errors import ServerError, APIError
from dotenv import load_dotenv

# ─── Load environment variables ────────────────────────────────────────────────
load_dotenv()  # Load .env file

# ─── Configuration ────────────────────────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL", "gemini-2.0-flash")

EXISTING_FILE = os.getenv("EXISTING_FILE", "existing_mapping.csv")
NEW_FILE = os.getenv("NEW_FILE", "CB.xlsx")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "updated_mapping.csv")
CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", "checkpoint_processed.txt")
ERROR_FILE = os.getenv("ERROR_FILE", "errors_unprocessed.txt")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "15"))
PAUSE_SECONDS = int(os.getenv("PAUSE_SECONDS", "60"))

# ─── Gemini client & search tool ──────────────────────────────────────────────
genai_client = genai.Client(api_key=API_KEY)
search_tool  = types.Tool(google_search=types.GoogleSearch())

# ─── Prompt template (from main.py) ──────────────────────────────────────────
PROMPT_TEMPLATE = """
╭────────────────────────  TASK  ────────────────────────╮
│  Step 1. Show concise reasoning.                       │
│  Step 2. Output JSON between <JSON>…</JSON> tags only. │
╰────────────────────────────────────────────────────────╯

────────────────────────  DEFINITIONS  ───────────────────
• **Startup** – Independent product company (own product, funding, team).
• **Gen-AI Startup** – Core value prop uses LLMs, diffusion, or agentic Gen-AI.
• **Layer** – Foundational / Infrastructure / Application.
• **Category** – {Content, Customer Service, Cyber Security, Data, DefTech,
  Dev Tools, Development, EdTech, Enterprise Platforms, Gaming, HealthTech,
  Knowledge Workers, LegalTech, Marketing, Note Taker, RFP, Safety, SalesTech,
  Science, HRTech, Consumer/Social}.
• **France link** – HQ, majority team, or founders clearly French.

────────────────────────  FORMAT  ───────────────────────
### REASONING
*Startup evidence*: 1-2 bullets  
*Gen-AI evidence*: 1-2 bullets  
*France evidence*: 0-2 bullets  
(Use short bullets ≤ 15 words each)

### RESULT
<JSON>
{{  "is_startup": …,
    "is_startup_confidence": …,
    "startup_rationale": "...",
    "is_gen_ai_startup": …,
    "is_gen_ai_startup_confidence": …,
    "gen_ai_rationale": "...",
    "layer": …,
    "layer_confidence": …,
    "category": …,
    "category_confidence": …,
    "is_linked_to_france": …,
    "is_linked_to_france_confidence": … }}
</JSON>

Rules  
• Confidence ≥ 60 only if boolean is true.  
• If publicly available info is truly insufficient → booleans false, layer/category null.  
• Favour *false* over *true* when evidence weak; reserve *null* only for no info.

────────────────────────  POSITIVE EXAMPLE  ──────────────
Input:
Name: Iktos
Website: https://iktos.ai/
Output:
### REASONING
*Startup evidence*: SaaS product page; Series A press release  
*Gen-AI evidence*: "Generative design of molecules"  
*France evidence*: Paris HQ  
### RESULT
<JSON>
{{"is_startup": true, "is_startup_confidence": 95,
  "startup_rationale":"SaaS+funding",
  "is_gen_ai_startup": true, "is_gen_ai_startup_confidence": 90,
  "gen_ai_rationale":"Gen design molecules",
  "layer":"Application","layer_confidence":85,
  "category":"Science","category_confidence":88,
  "is_linked_to_france": true,"is_linked_to_france_confidence":80}}
</JSON>

────────────────────────  NEGATIVE EXAMPLE  ──────────────
Input:
Name: A Kind of Magic
Website: https://www.akindofmagic.ai/
Output:
### REASONING
*Startup evidence*: Newsletter, no product  
*Gen-AI evidence*: n/a  
### RESULT
<JSON>
{{"is_startup": false,"is_startup_confidence":94,
  "startup_rationale":"Newsletter only",
  "is_gen_ai_startup": false,"is_gen_ai_startup_confidence":92,
  "gen_ai_rationale":"Media",
  "layer": null,"layer_confidence":0,
  "category": null,"category_confidence":0,
  "is_linked_to_france": false,"is_linked_to_france_confidence":70}}
</JSON>

────────────────────────  CLASSIFY  ───────────────────────
Name: {name}
Description (ID-check only – do NOT rely for classification): {description}

*First write the REASONING section, then the RESULT block.*
*Wrap the JSON object between <JSON> and </JSON> tags. Do not use markdown fences.*
This instruction in paramount, if you do not follow it the data you produce will not be usable.

This is a  correct format:
### RESULT
<JSON>
{{"is_startup": false,"is_startup_confidence":94,
  "startup_rationale":"Newsletter only",
  "is_gen_ai_startup": false,"is_gen_ai_startup_confidence":92,
  "gen_ai_rationale":"Media",
  "layer": null,"layer_confidence":0,
  "category": null,"category_confidence":0,
  "is_linked_to_france": false,"is_linked_to_france_confidence":70}}
</JSON>

This is not a correct format :
### RESULT
```json
{
  "is_startup": true,
  "is_startup_confidence": 80,
  "startup_rationale": "Website with product features",
  "is_gen_ai_startup": false,
  "is_gen_ai_startup_confidence": 50,
  "gen_ai_rationale": "No clear indication of GenAI usage, only generic AI.",
  "layer": null,
  "layer_confidence": 0,
  "category": "Enterprise Platforms",
  "category_confidence": 60,
  "is_linked_to_france": false,
  "is_linked_to_france_confidence": 50
}

The JSON must be valid RFC-8259: use double quotes for every key and string, no trailing commas.

Extra context : This batch of data comes from Crunchbase so the entries are very probably Startups.
"""

# ─── Helper function to classify companies ─────────────────────────────────────
def safe_classify_entity(name: str, description: str, max_retries: int = MAX_RETRIES) -> tuple[bool, dict]:
    """
    Returns (success_flag, result_dict).
    success_flag = True  → parsed JSON returned
    success_flag = False → nothing parsed; caller must NOT checkpoint this name
    """
    backoff = 2
    for attempt in range(1, max_retries + 1):
        try:
            prompt = (
                PROMPT_TEMPLATE
                + f"\n\nNow classify the following company:\n"
                + f"Name: {name}\n"
                + f"Description (ID-check only – do NOT rely on it): {description}\n"
            )
            resp = genai_client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(tools=[search_tool]),
            )
            raw = resp.text.strip()

            m = re.search(r"<JSON>(.*?)</JSON>", raw, re.S)
            if not m:
                raise ValueError("No <JSON> block found.")
            json_text = m.group(1).strip()
            result = json.loads(json_text)
            return True, result           # ← success
        except ServerError as e:          # transient 5xx
            code = getattr(e, "code", 503)
            if attempt < max_retries and code in (500, 502, 503, 504):
                wait = backoff * attempt
                print(f"[Retry {attempt}/{max_retries}] {code} – wait {wait}s")
                time.sleep(wait)
                continue
            print(f"❌  ServerError {code} for {name}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌  Parse error for {name}: {e}")
        except APIError as e:
            print(f"❌  API error for {name}: {e}")
        break                            # → exit retry loop

    # after all retries failed
    return False, {}                    # caller will log + retry later

def reprocess_errors(error_file=ERROR_FILE, max_per_run=50):
    """Process previously failed items from the error file."""
    if not os.path.exists(error_file):
        print(f"Error file {error_file} not found.")
        return
    
    print(f"Starting reprocessing from {error_file}...")
    
    # Read and deduplicate error file
    with open(error_file, "r", encoding="utf-8") as f:
        failed_names = list(dict.fromkeys(line.strip() for line in f if line.strip()))
    
    print(f"Found {len(failed_names)} unique failed items to reprocess.")
    
    # Limit number of reprocessed items per run
    to_process = failed_names[:max_per_run]
    successful = []
    still_failed = []
    
    # Load processed set to avoid reprocessing
    processed_set = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            processed_set = {line.strip() for line in f if line.strip()}
    
    # Filter out already processed items
    to_process = [name for name in to_process if name not in processed_set]
    if not to_process:
        print("All items in error file have already been processed. Nothing to do.")
        return []
    
    print(f"Will attempt to reprocess {len(to_process)} items.")
    
    # Get company details from source files
    source_dfs = []
    
    # Try to load Excel file
    try:
        excel_df = pd.read_excel(NEW_FILE)
        if "Company Name" not in excel_df.columns and excel_df.shape[1] >= 2:
            excel_df.columns = ["Company Name", "Description"] + list(excel_df.columns[2:])
        source_dfs.append(excel_df)
    except Exception as e:
        print(f"Warning: Could not load Excel file {NEW_FILE}: {e}")
    
    # Try to load CSV file
    try:
        csv_df = pd.read_csv(EXISTING_FILE)
        source_dfs.append(csv_df)
    except Exception as e:
        print(f"Warning: Could not load CSV file {EXISTING_FILE}: {e}")
    
    # Try to load the output file as well for comprehensive data
    try:
        if os.path.exists(OUTPUT_FILE):
            output_df = pd.read_csv(OUTPUT_FILE)
            source_dfs.append(output_df)
    except Exception as e:
        print(f"Warning: Could not load output file {OUTPUT_FILE}: {e}")
    
    if not source_dfs:
        print("Error: Could not load any source data files. Aborting.")
        return []
    
    # Combine all data sources
    combined_df = pd.concat(source_dfs, ignore_index=True)
    
    # Create a map for faster lookups
    company_map = {}
    if "Description" in combined_df.columns:
        company_map = {
            str(name): str(desc) for name, desc in 
            zip(combined_df["Company Name"], combined_df["Description"])
            if not pd.isna(name) and not pd.isna(desc)
        }
    
    records = []
    batch_counter = 0
    
    for i, name in enumerate(to_process):
        print(f"Processing {i+1}/{len(to_process)}: {name}")
        
        # Get description from data or use empty string
        desc = company_map.get(name, "")
        
        if not desc:
            print(f"Warning: No description found for {name}")
        
        success, result = safe_classify_entity(name, desc)
        
        if success:
            successful.append(name)
            print(f"✅ Successfully classified {name}")
            
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
                print(f"📊 Added to mapping: {name}")
            else:
                print(f"ℹ️ Not a Gen-AI startup: {name}")
            
            # Checkpoint successful reprocessing
            with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
                f.write(name + "\n")
        else:
            still_failed.append(name)
            print(f"❌ Failed to classify {name}")
        
        # Rate limiting
        batch_counter += 1
        if batch_counter >= BATCH_SIZE:
            print(f"⏸️  Pausing {PAUSE_SECONDS}s to respect rate limits...")
            time.sleep(PAUSE_SECONDS)
            batch_counter = 0
    
    # Update output file with newly processed items
    if records:
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            updated_df = pd.concat([existing_df, pd.DataFrame(records)], ignore_index=True)
            # Remove duplicates if any
            updated_df = updated_df.drop_duplicates(subset=["Company Name"], keep="first")
            updated_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Updated {OUTPUT_FILE} with {len(records)} new Gen-AI startups")
        except Exception as e:
            print(f"Error updating output file: {e}")
            # Save to a backup file instead
            pd.DataFrame(records).to_csv("reprocessed_results.csv", index=False)
            print(f"Saved results to reprocessed_results.csv instead")
    
    # Rewrite error file with remaining failures
    remaining_failures = list(set(failed_names) - set(successful))
    with open(error_file, "w", encoding="utf-8") as f:
        for name in remaining_failures:
            f.write(name + "\n")
    
    print(f"Reprocessing complete: {len(successful)} succeeded, {len(still_failed)} still failed")
    print(f"Added {len(records)} new Gen-AI startups to the mapping")
    return records

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reprocess failed items from error file")
    parser.add_argument("--max", type=int, default=50, help="Maximum number of items to process per run")
    parser.add_argument("--error-file", default=ERROR_FILE, help="Path to error file")
    args = parser.parse_args()
    
    reprocess_errors(error_file=args.error_file, max_per_run=args.max) 