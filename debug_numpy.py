#!/usr/bin/env python3
"""
Debug script to understand the numpy unpickling issue.
"""
import pickle
import numpy as np

# Check numpy version
print(f"NumPy version: {np.__version__}")

# Create a simple numpy array
arr = np.array([1, 2, 3])
print(f"Created array: {arr}")

# Pickle it
pickled = pickle.dumps(arr)
print(f"Pickled array size: {len(pickled)} bytes")

# Try to unpickle it normally
try:
    unpickled = pickle.loads(pickled)
    print(f"Successfully unpickled: {unpickled}")
except Exception as e:
    print(f"Error unpickling: {e}")

# Check what the unpickler sees
class DebugUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print(f"find_class called: module={module}, name={name}")
        return super().find_class(module, name)

print("\n--- Testing with debug unpickler ---")
with open('/tmp/test_array.pkl', 'wb') as f:
    pickle.dump(arr, f)

with open('/tmp/test_array.pkl', 'rb') as f:
    unpickler = DebugUnpickler(f)
    result = unpickler.load()
    print(f"Result: {result}")

# Check what happens with a dict containing numpy array
data = {'array': arr, 'value': 42}
print("\n--- Testing with dict containing numpy array ---")
with open('/tmp/test_dict.pkl', 'wb') as f:
    pickle.dump(data, f)

with open('/tmp/test_dict.pkl', 'rb') as f:
    unpickler = DebugUnpickler(f)
    result = unpickler.load()
    print(f"Result keys: {list(result.keys())}")
    print(f"Array type: {type(result['array'])}")

# Check the actual brain file if it exists
import os
brain_path = "urcm_identity.pkl"
if os.path.exists(brain_path):
    print(f"\n--- Checking brain file: {brain_path} ---")
    with open(brain_path, 'rb') as f:
        # Try to read the first few bytes to see if it's a pickle
        header = f.read(100)
        print(f"First 100 bytes: {header}")
        f.seek(0)
        
        # Try to unpickle with debug
        unpickler = DebugUnpickler(f)
        try:
            result = unpickler.load()
            print(f"Successfully unpickled brain data")
            print(f"Keys in brain_data: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        except Exception as e:
            print(f"Error unpickling brain file: {e}")
else:
    print(f"\nBrain file not found: {brain_path}")
