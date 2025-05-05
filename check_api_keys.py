import os
import time
import google.generativeai as genai
from google.api_core import exceptions as api_exceptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys
API_KEY = os.getenv("GEMINI_API_KEY")
api_keys_raw = os.getenv("GEMINI_API_KEYS", API_KEY)
API_KEYS = [k.strip() for k in api_keys_raw.split(',') if k.strip()]

def check_key(api_key, index):
    """Test if an API key is valid by sending a simple request"""
    try:
        print(f"Testing API key {index+1}...", end="", flush=True)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Send a minimal request to test the key
        start_time = time.time()
        response = model.generate_content("Hello")
        elapsed = time.time() - start_time
        
        # Check if we got a valid response
        if response and hasattr(response, 'text'):
            print(f" ✅ Valid! (Response time: {elapsed:.2f}s)")
            return True
        else:
            print(f" ❌ Invalid (No response text)")
            return False
    except api_exceptions.InvalidArgument:
        print(f" ❌ Invalid API key format")
        return False
    except api_exceptions.PermissionDenied:
        print(f" ❌ API key doesn't have permission")
        return False
    except api_exceptions.ResourceExhausted:
        print(f" ⚠️ Rate limited (key might be valid but exceeded quota)")
        return True  # Consider rate-limited keys as valid
    except Exception as e:
        print(f" ❌ Error: {str(e)}")
        return False

def main():
    print(f"Found {len(API_KEYS)} API key(s) to test")
    
    valid_keys = 0
    for i, key in enumerate(API_KEYS):
        if check_key(key, i):
            valid_keys += 1
        # Brief pause between tests to avoid rate limiting
        time.sleep(1)
    
    print("\nSummary:")
    print(f"- Total keys tested: {len(API_KEYS)}")
    print(f"- Valid keys: {valid_keys}")
    
    if valid_keys == 0:
        print("\n❌ ERROR: No valid API keys found! Please check your .env file.")
        return 1
    elif valid_keys < len(API_KEYS):
        print("\n⚠️ WARNING: Some API keys appear to be invalid.")
    else:
        print("\n✅ All API keys are valid!")
    
    print("\nRecommended configuration in .env file:")
    print(f"MAX_WORKERS={min(valid_keys, 4)}")
    print(f"QUOTA_PER_MINUTE={valid_keys * 15}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 