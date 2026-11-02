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
| `cleansplit.py` | Таблица комплексов CleanSplit ∩ R1, фильтры меток, сплит по ID |
| `esm_converter.py` | Предрасчёт эмбеддингов белков ESM-2 → `esm_embeddings.pt` |

## Данные и протокол

Протокол: **PDBbind v2020.R1 + CleanSplit** (Graber et al., 2025). Train = CleanSplit train ∩ R1, валидация = случайные 15 % train (`val_mode = "random"`; можно переключить на фолды GEMS: `val_mode = "gems_fold"`, `fold = 0..4`), тест = CASF-2016 (основной), а также CASF-2016 indep, CASF-2013 и CASF-2013 indep. Метки берутся из индекса R1, pK = −log10(Kd/Ki/IC50).

Данные в репозиторий **не входят** (лицензия PDBbind и размер). Что нужно положить в рабочую папку:

- `PDBbind_v2020R1/` — распакованный пакет «Protein-ligand complex structures» из PDBbind v2020.R1 (PDBbind+, нужна регистрация). Вложенность папок не важна: файлы `<id>_ligand.mol2|sdf` и `<id>_pocket.pdb` ищутся рекурсивно;
- индексный файл с аффинностями из той же загрузки. Путь задаётся в `config.r1_index_file`; если его там нет, скрипт ищет `INDEX_general_PL*` внутри `PDBbind_v2020R1/`;
- `esm_embeddings.pt` (нужен только для `full` и `esm_only`), создаётся скриптом `esm_converter.py`.

Списки CleanSplit уже лежат в `splits/cleansplit/` (см. README там).

`data/`, `processed/` и `outputs/` создаются автоматически и игнорируются git.

## Запуск

```bash
pip install -r requirements.txt
python cleansplit.py      # таблица data/complexes.csv + отчёт о покрытии
python esm_converter.py   # один раз, если нужен ESM
python main.py
```

`cleansplit.py` нужно перезапускать, если поменялись структуры или индексный файл. Фильтры меток (`drop_censored_labels`, `keep_approx_labels`) `main.py` применяет сам, поэтому после их изменения ничего пересобирать не нужно: графы строятся для всех комплексов с меткой. Если метки в `complexes.csv` изменились, `main.py` заметит расхождение с кэшем графов и попросит удалить `processed/graphs_<tag>*`.

## Что сохраняется в `outputs/<model>_<val15|fold k>_seed<s>/`

| Файл | Содержимое |
|---|---|
| `split_info.json` | Настройки сплита, sha256 файлов сплита и `complexes.csv`, списки train/val/test ID, исключённые ID с причиной (`no_structure`, `no_label`, `censored_label`, `approx_label`, `graph_failed:<причина>`) |
| `test_metrics.pt` | Метрики по каждому тестовому набору |
| `test_predictions.csv` | Предсказания по комплексам: `test_set, pdb_id, y_true, y_pred` |
| `best_model_rmse.pt`, `model_final_pKd.pt`, `y_stats.pt`, `training_history.pt` | Как раньше |

Причины, по которым графы не построились, лежат в `processed/graphs_<tag>_index.json`.
