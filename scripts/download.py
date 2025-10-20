from scripts.utils import download_file, extract_7z

url = "http://leakmirror.wikileaks.org/file/straw-glass-and-bottle/afg-war-diary.csv.7z"
local_filename = "afg-war-diary.csv.7z"
extract_dir = "../data/raw/wikileaks/afg-war-diary"

if download_file(url, local_filename):
    extract_7z(local_filename, extract_dir)