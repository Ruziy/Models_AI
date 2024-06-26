import os
import shutil
import random
import string

source_folder = "work_with_book/open_code_TZ/create_datasets/archive"
target_folder = "work_with_book/open_code_TZ/create_datasets/images"

def copy_images(source_folder, target_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            # Генерируем случайное имя файла с явным указанием кодировки
            random_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            target_path = os.path.join(target_folder, f"{random_name}{ext}")
            try:
                shutil.copy2(source_path, target_path)
            except UnicodeDecodeError:
                # Обработка ошибки кодировки при копировании
                print(f"Ошибка при копировании файла: {source_path}")

# Создаем папку для изображений, если она еще не существует
os.makedirs(target_folder, exist_ok=True)

# Копируем изображения
copy_images(source_folder, target_folder)
