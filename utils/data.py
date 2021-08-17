import glob, os

def remove_processed_data():
    files = glob.glob(r'data/processed/*')
    for f in files:
        os.remove(f)
    print("Files removed.")
