import torch
import pandas as pd
import esm
from tqdm import tqdm

# 1. Загрузка модели ESM-2
print("Загрузка ESM-2 модели...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval() 

if torch.cuda.is_available():
    model = model.cuda()

def get_esm_embedding(sequence):
    # Если цепочек несколько, усредняем их эмбеддинги
    sub_seqs = sequence.split(':')
    all_embeddings = []
    
    for s in sub_seqs:
        # ESM-2 не любит слишком длинные или пустые строки
        s = s[:1022] # Ограничение по длине контекста
        data = [("protein", s)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)
        
        # Извлекаем эмбеддинг (усредняем по длине последовательности)
        token_representations = results["representations"][6]
        # Берем среднее, исключая токены начала и конца (CLS, EOS)
        seq_representation = token_representations[0, 1 : len(s) + 1].mean(0)
        all_embeddings.append(seq_representation.cpu())
    
    # Возвращаем среднее по всем цепочкам
    return torch.stack(all_embeddings).mean(0)

# 2. Основной цикл
df = pd.read_csv("protein_sequences.csv")
esm_dict = {}

print(f"Начинаю генерацию для {len(df)} белков...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        esm_dict[row['pdb_id']] = get_esm_embedding(row['sequence'])
    except Exception as e:
        print(f"Ошибка в {row['pdb_id']}: {e}")

# 3. Сохранение
torch.save(esm_dict, "esm_embeddings.pt")
print("✅ Готово! Файл 'esm_embeddings.pt' создан.")