# Binding affinity prediction: GNN + quantum descriptors

Код baseline-модели для диссертации: предсказание аффинности связывания белок–лиганд (PDBbind) с помощью графовой нейросети (gated GATv2) и опциональной вариационной квантовой головы (VQC, PennyLane).

## Файлы

| Файл | Назначение |
|---|---|
| `config.py` | Пути и гиперпараметры |
| `dataset.py` | Построение графов комплексов (лиганд + карман) |
| `Graph_Attention_Layer.py` | Слой графового внимания |
| `model.py` | Модель: GNN-энкодер, MLP / VQC голова, ESM через FiLM |
| `trainer.py` | Цикл обучения и валидации |
| `utils.py` | Метрики и вспомогательные функции |
| `main.py` | Точка входа: обучение и тест |
| `esm_converter.py` | Предрасчёт эмбеддингов белков ESM-2 → `esm_embeddings.pt` |

## Данные

Данные в репозиторий **не входят** (лицензия PDBbind и размер). Ожидается, что в рабочей папке лежат:

- `refined_set/` — PDBbind 2016 refined set;
- индексный файл с аффинностями и список CASF-2016 core (пути в `config.py`);
- `esm_embeddings.pt` — создаётся скриптом `esm_converter.py`.

`processed/` и `outputs/` создаются автоматически и тоже игнорируются git.

## Запуск

```bash
pip install -r requirements.txt
python esm_converter.py   # один раз, если нужен ESM
python main.py
```
