import os, re, json, time
import pandas as pd
import google.generativeai as genai
from google.generativeai import types
from google.api_core import exceptions as api_exceptions
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

CATEGORIES = [
    "Content","Customer Service","Cyber Security","Data","DefTech","Dev Tools",
    "Development","EdTech","Enterprise Platforms","Gaming","HealthTech",
    "Knowledge Workers","LegalTech","Marketing","Note Taker","RFP","Safety",
    "SalesTech","Science","HRTech","Consumer/Social"
]

# ─── Gemini client & search tool ──────────────────────────────────────────────
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL)

# ─── Prompt template (unchanged, brace-safe) ─────────────────────────────────
PROMPT_TEMPLATE = """
You are a specialized AI trained to classify French startups leveraging Generative AI. Your task is to analyze companies and output structured data about them.

IMPORTANT: Take a step-by-step approach in your analysis. First, thoroughly research and reason about each criterion before making a determination.

╭────────────────────────  TASK  ────────────────────────╮
│  Step 1. THINK through your analysis carefully.        │
│  Step 2. Show detailed reasoning with evidence.        │
│  Step 3. Output JSON between <JSON>…</JSON> tags.      │
╰────────────────────────────────────────────────────────╯

────────────────────────  CONTEXT  ───────────────────────
This batch of data comes from Crunchbase, which focuses on startups and technology companies. This increases the prior probability that the companies are indeed startups, but you should still evaluate each case carefully.

France has a growing Gen-AI ecosystem with hubs in Paris, Lyon, and other major cities. The French government has been actively supporting AI development through initiatives like "France 2030" and "AI for Humanity." Many French Gen-AI startups have international presence while maintaining French roots.

────────────────────────  DEFINITIONS  ───────────────────
• **Startup** – Independent product company (own product, funding, team). Look for evidence of product offerings, funding rounds, team size, founding date, and independence from larger corporations.

• **Gen-AI Startup** – Core value proposition uses Large Language Models, diffusion models, or other generative AI technologies. Look for specific mentions of LLMs, diffusion models, generative AI, or specific applications like text generation, image generation, code generation, etc.
  - True Gen-AI startups build products fundamentally powered by generative models
  - Not Gen-AI: Companies that merely use basic AI/ML techniques, analytics, or just mention AI without generative capabilities

• **Layer** – Choose one:
  - Foundational: Creates core Gen-AI models/technologies (like developing new LLMs)
  - Infrastructure: Provides tools/platforms for others to build Gen-AI applications (APIs, model hosting)
  - Application: Uses Gen-AI to solve specific problems for end-users (most common category)

• **Category** – Choose the most appropriate:
  {Content, Customer Service, Cyber Security, Data, DefTech, Dev Tools, Development, EdTech, Enterprise Platforms, Gaming, HealthTech, Knowledge Workers, LegalTech, Marketing, Note Taker, RFP, Safety, SalesTech, Science, HRTech, Consumer/Social}

• **France link** – HQ in France, majority of team in France, or founders clearly French. Look for location information, team background, language of website, etc.

────────────────────────  CONFIDENCE SCORING  ───────────────────────
• 90-100: Extremely high confidence with multiple strong pieces of evidence
• 80-89: High confidence with clear evidence
• 70-79: Good confidence with supportive evidence
• 60-69: Moderate confidence with some evidence
• 50-59: Low confidence with limited evidence (use for "false" determinations)
• 0: Used only when field is null (not applicable)

────────────────────────  REASONING APPROACH  ───────────────────────
For each company, follow these steps in your analysis:

1. STARTUP ANALYSIS:
   - What specific products or services does the company offer?
   - Is there evidence of funding, team structure, or independence?
   - What stage is the company in (early-stage, growth, established)?

2. GEN-AI ASSESSMENT:
   - Does the company explicitly mention using generative AI technologies?
   - What specific Gen-AI technologies or applications are mentioned?
   - Is Gen-AI central to their value proposition or just a supporting feature?
   - Distinguish between true Gen-AI (generative models) vs. general AI/ML

3. LAYER & CATEGORY DETERMINATION:
   - Based on their offerings, which layer best describes their position?
   - Which category most accurately captures their focus area?

4. FRANCE CONNECTION:
   - What evidence exists for a French connection?
   - Is the company headquartered in France?
   - Are the founders or team members French?
   - Is the website available in French or targeting French markets?

────────────────────────  HANDLING EDGE CASES  ───────────────────────
• Consulting firms: If a company primarily offers consulting services using Gen-AI rather than an actual Gen-AI product, classify as NOT a Gen-AI startup.
• Pre-launch companies: If a company is in stealth mode or pre-product, base your assessment on their stated intentions and progress.
• AI companies without clear Gen-AI focus: Companies using traditional AI/ML without generative capabilities are NOT Gen-AI startups.
• Companies with minimal online presence: Use available information to make best judgment, but lean toward lower confidence scores.

────────────────────────  FORMAT  ───────────────────────
### REASONING
*Startup evidence*: 2-3 detailed bullet points with specific evidence.  
*Gen-AI evidence*: 2-3 detailed bullet points with specific technologies and applications.  
*Layer reasoning*: 1-2 bullet points explaining why this layer classification fits.  
*Category reasoning*: 1-2 bullet points explaining the category choice.  
*France evidence*: 1-2 bullet points about French connections.  

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

Rules:  
• Confidence ≥ 60 only if boolean is true.  
• If information is truly insufficient → booleans false, layer/category null.  
• Favour *false* over *true* when evidence is weak; reserve *null* only for no info.
• ALWAYS wrap the JSON between <JSON> and </JSON> tags.
• ALWAYS use double quotes for JSON strings.
• NEVER include trailing commas in JSON.

────────────────────────  POSITIVE EXAMPLE  ──────────────
Input:
Name: Iktos
Website: https://iktos.ai/
Output:
### REASONING
*Startup evidence*: 
- Company has a SaaS product page with detailed offerings for molecule design
- Press releases mention Series A funding round, indicating venture backing
- Team page shows multiple employees with specialized roles

*Gen-AI evidence*: 
- Explicitly mentions "Generative design of molecules" on homepage
- Uses deep generative models to create novel molecular structures
- Offers AI-based retrosynthesis planning using generative models

*Layer reasoning*:
- Functions as an Application layer product as it delivers specific Gen-AI solutions for end users in pharmaceutical industry

*Category reasoning*:
- Clearly fits in Science category due to focus on molecule design and drug discovery

*France evidence*:
- Headquarters listed as Paris, France
- Founding team includes French researchers

### RESULT
<JSON>
{{"is_startup": true, "is_startup_confidence": 95,
  "startup_rationale":"SaaS product with Series A funding",
  "is_gen_ai_startup": true, "is_gen_ai_startup_confidence": 90,
  "gen_ai_rationale":"Uses generative models for molecule design",
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
*Startup evidence*: 
- Only offers a newsletter about AI trends, no actual product identified
- No information about funding, team size, or company structure
- Functions more as a media outlet than a product company

*Gen-AI evidence*: 
- Discusses AI in general but doesn't offer any Gen-AI products or services
- No evidence of developing or applying generative AI technologies
- Appears to be content-focused rather than technology-focused

*Layer reasoning*:
- N/A - Not a Gen-AI startup so layer classification doesn't apply

*Category reasoning*:
- N/A - Not a Gen-AI startup so category classification doesn't apply

*France evidence*:
- No clear connection to France in available information
- Website and content appear to be in English with no French location mentioned

### RESULT
<JSON>
{{"is_startup": false,"is_startup_confidence":94,
  "startup_rationale":"Newsletter only, no product offering",
  "is_gen_ai_startup": false,"is_gen_ai_startup_confidence":92,
  "gen_ai_rationale":"Content about AI but no Gen-AI technology",
  "layer": null,"layer_confidence":0,
  "category": null,"category_confidence":0,
  "is_linked_to_france": false,"is_linked_to_france_confidence":70}}
</JSON>

────────────────────────  CLASSIFY  ───────────────────────
Name: {name}
Description (ID-check only – do NOT rely on it): {description}

Remember: Take your time to analyze thoroughly, provide detailed reasoning, and ALWAYS wrap your JSON response in <JSON> tags.
"""

# ─── Helper: classify with retries ────────────────────────────────────────────
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
            retry_delay = 60  # Default to 60 seconds if not specified
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


# ─── Load datasets ───────────────────────────────────────────────────────────
existing_df = pd.read_csv(EXISTING_FILE)
new_df      = pd.read_excel(NEW_FILE)        # Name + Description

# keep first two columns only
new_df = new_df.iloc[:, :2]
new_df.columns = ["Company Name", "Description"]

todo_df = new_df[~new_df["Company Name"].isin(existing_df["Company Name"])]
print(f"Found {len(todo_df)} new companies to process")

# ─── Checkpoint ───────────────────────────────────────────────────────────────
processed_set = set()
if os.path.exists(CHECKPOINT_FILE):
    processed_set = {line.strip() for line in open(CHECKPOINT_FILE, encoding="utf-8")}

# ─── Main loop ────────────────────────────────────────────────────────────────
records = []
batch_counter = 0  # counts calls since last sleep

for _, row in todo_df.iterrows():
    name = row["Company Name"]
    desc = row["Description"]

    # Skip if already processed in a previous run
    if name in processed_set:
        continue

    success, result = safe_classify_entity(name, desc)

    if not success:
        # Keep a log of items that still need processing; do NOT checkpoint the name
        with open(ERROR_FILE, "a", encoding="utf-8") as err:
            err.write(name + "\n")
        continue  # move to next company

    # ► Successful parse → checkpoint immediately
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(name + "\n")
    processed_set.add(name)

    # Manual pause every X successful API calls
    batch_counter += 1
    if batch_counter >= BATCH_SIZE:
        print(f"⏸️  Pausing {PAUSE_SECONDS}s to respect rate limits …")
        time.sleep(PAUSE_SECONDS)
        batch_counter = 0

    # Collect Gen-AI startups for the final mapping
    if result.get("is_startup") and result.get("is_gen_ai_startup"):
        records.append(
            {
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
        )

# ─── Save output ─────────────────────────────────────────────────────────────
updated_df = (
    pd.concat([existing_df, pd.DataFrame(records)], ignore_index=True)
    if records else existing_df
)
updated_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅  Added {len(records)} Gen-AI startups → {OUTPUT_FILE}")
print(f"🔖  Progress checkpoint saved to {CHECKPOINT_FILE}")
