# Распознавание трещин на изображениях

Проект на базе Ultralytics YOLO решает две задачи: классифицирует изображение как `cracked` / `uncracked`, а затем строит маску найденной трещины.

## Подготовка

Запускайте команды из корня проекта. Нужен Python 3.12+; для обучения желательно использовать CUDA-совместимую видеокарту.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Данные классификации должны находиться в `dataset/{train,val,test}/{cracked,uncracked}`, исходные изображения и маски сегментации — в `segmentation/{train,val,test}/{images,masks}`. Пути к данным сегментации определяются относительно `yolo_seg/data.yaml`, поэтому после клонирования проекта менять их не требуется.

## Воспроизведение

Подготовка разметки и обучение моделей:

```powershell
python train.py
python scripts/prepare_yolo_seg.py
python train_yolo_seg.py
```

Ultralytics автоматически использует доступный GPU, а при его отсутствии переключается на CPU. Обучение на CPU будет значительно дольше.

Оценка и сохранение примеров сегментации:

```powershell
python evaluate.py
python evaluate_segmentation.py
python visualize_segmentation.py
```

Метрики классификации и сегментации выводятся в консоль. Графики обучения сохраняются в `runs/`, а десять сравнений масок — в `evaluation/visualizations/`.

Для демонстрации готовых весов поместите проверяемое изображение в `test.jpg` и выполните:

```powershell
python pipeline.py
```

В консоли появятся класс и уверенность модели; для изображения с трещиной откроются окна с оригиналом, бинарной маской и наложением маски. Готовые веса уже лежат в `runs/classify/train-3/weights/best.pt` и `runs/segmentation/yolov8n_seg-2/weights/best.pt`.
