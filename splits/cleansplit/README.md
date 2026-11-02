# PDBbind CleanSplit (ID lists)

Файлы скопированы без изменений из репозитория GEMS (Graber et al., *Nat. Mach. Intell.* 7, 1713–1725, 2025), папка `PDBbind_data/`:
https://github.com/camlab-ethz/GEMS (лицензия MIT, © 2024 David Graber). Скачаны 2026-11-02 из ветки `main`; контрольные суммы в `SHA256SUMS`.

Здесь только списки PDB ID и метки, структур PDBbind нет.

| Файл | Содержимое |
|---|---|
| `PDBbind_data_split_cleansplit.json` | `train` (16 908), `casf2016` (285), `casf2016_indep` (144), `casf2013` (195), `casf2013_indep` (101). `*_indep` являются подмножествами соответствующих CASF. |
| `PDBbind_cleansplit_train_val_split_f{0..4}.json` | 5 фолдов GEMS: `train` / `validation` (≈ 80/20). Все фолды покрывают одни и те же 16 491 ID; валидационные части не пересекаются. В фолдах нет меток `>` (205 шт.) и ещё 212 ID из `train`. |
| `PDBbind_data_dict.json` | Метки PDBbind v2020 (оригинальные): `log_kd_ki`, `precision`, `Kd`/`Ki`/`IC50`, `resolution`, `dataset`. Используется как запасной источник меток и для сверки с R1. |

Матрицы попарного сходства (TM-score, Tanimoto, RMSD позы) в репозиторий не входят: они лежат на Zenodo (https://doi.org/10.5281/zenodo.15482796) и понадобятся для анализа ошибок по сходству с train.
