# Распознавание трещин на изображениях

Проект на базе Ultralytics YOLO решает две задачи: классифицирует изображение как `cracked` / `uncracked`, а затем строит маску найденной трещины.

## Подготовка

Запускайте команды из корня проекта. Нужен Python 3.10+; для обучения желательно использовать CUDA-совместимую видеокарту.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install ultralytics opencv-python numpy matplotlib pandas
```

Данные классификации должны находиться в `dataset/{train,val,test}/{cracked,uncracked}`, исходные изображения и маски сегментации — в `segmentation/{train,val,test}/{images,masks}`. При переносе проекта измените поле `path` в `yolo_seg/data.yaml` на абсолютный путь к папке `yolo_seg`.

## Воспроизведение

Подготовка разметки и обучение моделей:

```powershell
python train.py
python scripts/prepare_yolo_seg.py
python train_yolo_seg.py
```

Скрипты используют GPU (`device=0`). Для запуска на CPU замените это значение в `train.py` и `train_yolo_seg.py` на `device="cpu"` (обучение будет значительно дольше).

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

В консоли появятся класс и уверенность модели; для изображения с трещиной откроются окна с оригиналом, бинарной маской и наложением маски. Готовые веса уже лежат в `runs/classify/train-3/weights/best.pt` и `runs/segment/runs/segmentation/yolov8n_seg-2/weights/best.pt`.
