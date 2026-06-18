
import urllib.request
import os
import sys

def download_file(url, filename):
    print(f"⬇️ Downloading {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
            
        # Basic cleanup for Gutenberg headers/footers
        # Usually starts with "*** START OF THE PROJECT GUTENBERG EBOOK"
        # Ends with "*** END OF THE PROJECT GUTENBERG EBOOK"
        
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        
        start_idx = data.find(start_marker)
        end_idx = data.find(end_marker)
        
         if start_idx != -1:
             data = data[start_idx + len(start_marker):].lstrip('\n')
         if end_idx != -1:
             data = data[:end_idx]
             
         # Save
         with open(filename, 'w', encoding='utf-8') as f:
             f.write(data)
             f.flush()
             
         size_kb = len(data) / 1024
         print(f"✅ Saved to {filename} ({size_kb:.2f} KB)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    url = "https://www.gutenberg.org/files/1661/1661-0.txt" # Sherlock Holmes
    output = "training_data/sherlock.txt"
    download_file(url, output)
