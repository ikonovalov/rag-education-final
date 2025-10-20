import os

import py7zr
import requests


def download_file(url, local_path):
    """Скачивает файл по указанному URL и сохраняет его локально."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Проверка на ошибки HTTP
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Файл успешно скачан: {local_path}")
        return True
    except requests.RequestException as e:
        print(f"Ошибка при скачивании файла: {e}")
        return False

def extract_7z(file_path, extract_dir):
    """Распаковывает 7z файл в указанную директорию."""
    try:
        if not os.path.exists(file_path):
            print(f"Файл не найден: {file_path}")
            return False

        # Создаем директорию для распаковки, если она не существует
        os.makedirs(extract_dir, exist_ok=True)

        with py7zr.SevenZipFile(file_path, mode='r') as archive:
            archive.extractall(path=extract_dir)
        print(f"Файл успешно распакован в: {extract_dir}")
        return True
    except py7zr.exceptions.SevenZipException as e:
        print(f"Ошибка при распаковке файла: {e}")
        return False