import ee

print("Starting Google Earth Engine Authentication Flow...")
try:
    # This will open your browser to log in
    ee.Authenticate()
    print("\n[SUCCESS] Authentication completed. You can now run the extraction scripts.")
except Exception as e:
    print(f"\n[ERROR] Authentication failed: {e}")
    print("Please ensure you have an active internet connection and a Google account.")
