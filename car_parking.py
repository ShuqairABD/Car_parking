# Глобальная переменная для пути к весам YOLO
yolo_weights_path = "bestyolo11.pt"

# Функция для обработки загруженного файла
# Эта функция обновляет глобальный путь к YOLO весам после загрузки нового файла
def upload_yolo_weights(file):
    global yolo_weights_path
    yolo_weights_path = file
    return f"YOLO веса успешно обновлены: {file}"

# Функции для работы с базой данных

# Инициализация базы данных: создание файла базы данных, если он не существует
def initialize_database(filename='database.txt'):
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            pass

# Загрузка базы данных: чтение данных из файла базы данных
def load_database(filename='database.txt'):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return [line.strip() for line in file.readlines()]
    return []

# Добавление номера в базу данных: проверяет, существует ли номер, и добавляет его, если он отсутствует
def add_to_database(plate, filename='database.txt'):
    plates = load_database(filename)
    if plate not in plates:
        with open(filename, 'a') as file:
            file.write(plate + '\n')
        return f"Номер {plate} добавлен в базу данных."
    else:
        return f"Номер {plate} уже существует в базе данных."

# Распознавание номерного знака
# Эта функция принимает изображение, выполняет OCR и возвращает распознанный номер
def recognize_license_plate(image):
    reader = easyocr.Reader(['en'])  # Используется EasyOCR для распознавания текста
    image_plate = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразование изображения в RGB

    # Распознавание текста с изображения
    result = reader.readtext(image_plate, detail=0)
    recognized_plate = ''.join(result).replace(' ', '').upper()  # Удаление пробелов и преобразование в верхний регистр
    cleaned_plate = recognized_plate[:13]  # Ограничение длины распознанного текста

    # Форматирование номера
    formatted_plate = ""
    if "RUS" in cleaned_plate:
        cleaned_plate = cleaned_plate.replace("RUS", "")
        formatted_plate = (
            f"{cleaned_plate[:1]} {cleaned_plate[1:4]} "
            f"{cleaned_plate[4:6]} {cleaned_plate[7:]}".strip() + " RUS"
        )
    else:
        formatted_plate = (
            f"{cleaned_plate[:1]} {cleaned_plate[1:4]} "
            f"{cleaned_plate[4:6]} {cleaned_plate[7:]}".strip()
        )

    return formatted_plate

# YOLO11 для анализа парковочных мест
# Эта функция использует YOLO для анализа изображения парковочного места
def analyze_parking(image):
    model = YOLO(yolo_weights_path)  # Загружается модель YOLO с указанными весами
    results = model.predict(image, save=True, save_txt=False)  # Выполняется предсказание

    free_slots = 0  # Количество свободных мест
    occupied_slots = 0  # Количество занятых мест
    annotated_image = None

    # Обработка результатов
    for result in results:
        annotated_image = result.plot()  # Создание аннотированного изображения
        for box in result.boxes:
            cls = int(box.cls)
            if cls == 0:  # Класс 0 — занятое место
                occupied_slots += 1
            else:  # Класс 1 — свободное место
                free_slots += 1

    formatted_message = display_results(free_slots, occupied_slots)
    return annotated_image, formatted_message

# Форматирование результатов парковки
# Генерирует HTML-код для отображения результатов
def display_results(free_slots, occupied_slots):
    total_slots = free_slots + occupied_slots  # Общее количество мест
    result_message = f"""
    <div style="
        font-family: Arial, sans-serif;
        text-align: center;
        background-color: #f9f9f9;
        padding: 20px;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        max-width: 400px;
        margin: 0 auto;">
        <h2 style="color: #4CAF50; margin: 10px 0;">Добро пожаловать на парковку!</h2>
        <p style="font-size: 18px; margin: 5px 0;">✅ Свободные места: <strong>{free_slots}</strong></p>
        <p style="font-size: 18px; margin: 5px 0;">❌ Занятые места: <strong>{occupied_slots}</strong></p>
        <p style="font-size: 18px; margin: 5px 0;">ℹ️ Всего мест: <strong>{total_slots}</strong></p>
    </div>
    """
    return result_message

# Обработка изображений и базы данных
# Выполняет распознавание номера, добавление в базу данных и анализ парковочного места
def process_license_and_parking(license_image, parking_image, add_to_db):
    license_plate = recognize_license_plate(license_image)  # Распознавание номера

    # Обработка базы данных
    if add_to_db:
        db_result = add_to_database(license_plate)
    else:
        db_result = f"Номер {license_plate} не добавлен в базу данных."

    # Анализ парковочного места
    parking_annotated_image, parking_result = analyze_parking(parking_image)
    return license_plate, db_result, parking_annotated_image, parking_result

# Настройка Gradio интерфейса с аккордионом
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='font-size: 30px; font-style: italic;'><strong>Система для распознавания номерных знаков и анализа парковочных мест</strong></h1>")

    # Раздел для загрузки весов YOLO в аккордион
    with gr.Accordion("Настройка YOLO весов", open=False):
        with gr.Row():
            yolo_weights_upload = gr.File(type="filepath", label="Загрузить веса YOLO")
            upload_button = gr.Button("Обновить веса")
            upload_result = gr.Textbox(label="Результат обновления весов")

        upload_button.click(
            upload_yolo_weights,
            inputs=[yolo_weights_upload],
            outputs=[upload_result]
        )

    # Раздел для обработки изображений
    with gr.Row():
        license_input = gr.Image(type="numpy", label="Изображение номерного знака")
        parking_input = gr.Image(type="numpy", label="Изображение парковочного места")

    add_to_db_checkbox = gr.Checkbox(label="Добавить номер в базу данных?", value=False)

    submit_button = gr.Button("Обработать")

    license_output = gr.Textbox(label="Распознанный номер")
    db_output = gr.Textbox(label="Результат базы данных")
    parking_output_image = gr.Image(label="Обработанное изображение парковочного места")
    parking_output_text = gr.HTML(label="Результат анализа парковки")

    submit_button.click(
        process_license_and_parking,
        inputs=[license_input, parking_input, add_to_db_checkbox],
        outputs=[license_output, db_output, parking_output_image, parking_output_text]
    )

# Инициализация базы данных и запуск интерфейса
initialize_database()
demo.launch()
