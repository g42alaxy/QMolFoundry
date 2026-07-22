<h1 align="center">
    <img width="300" height="auto" src="https://github.com/user-attachments/assets/492454de-726b-47c0-8399-1a7edfc22ef1"/>
</h1>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6.19-F97316?logo=gradio&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-2AB6C8)
![Qiskit](https://img.shields.io/badge/Qiskit-2.4-6929C4?logo=qiskit&logoColor=white)
![Qiskit Aer](https://img.shields.io/badge/Qiskit_Aer-0.17-6929C4)
![RDKit](https://img.shields.io/badge/RDKit-2026.03-009999)
![NumPy](https://img.shields.io/badge/NumPy-2.4-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-11557C)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-linted-D7FF64?logo=ruff&logoColor=black)
![pytest](https://img.shields.io/badge/pytest-tested-0A9EDC?logo=pytest&logoColor=white)

<br> 
<a href="https://huggingface.co/spaces/g42alaxy/QMolFoundry">Hugging Face Space демонстрация</a> 
<br>

</div>



### 1. Что внутри проекта:

Основна проекта — это Gradio-showcase веб-приложение, которое поднимает все семейство моделей [Hybrid Quantum Cycle GAN](https://ieeexplore.ieee.org/abstract/document/10556803)
(`generator × cycle`) на едином датасете (QM9 + PC9). Позволяя генерировать молекулы как полностью классические, так и полностью квантово с помощью визуального интерфейса. При этом никакой внешней загрузки весов или IBM-аккаунта не требуется, так как всё необходимое лежит в
репозитории.

**Конфигурации моделей** (`generator` × `cycle`):

|                    | Без cycle-компоненты          | Classical cycle              | Quantum cycle                     |
|--------------------|-------------------------------|------------------------------|-----------------------------------|
| **Classic**        | MolGAN                        | Cycle MolGAN                 | Hybrid Cycle MolGAN               |
| **VVRQ** (quantum) | HQ-MolGAN (VVRQ)              | HQ Cycle MolGAN (VVRQ)       | Hybrid-Cycle HQ-MolGAN (VVRQ)     |
| **EFQ** (quantum)  | HQ-MolGAN (EFQ)               | HQ Cycle MolGAN (EFQ)        | Hybrid-Cycle HQ-MolGAN (EFQ)      |

> Для генерации используется только генератор `G`. 
> В зависимости от конфигурации, поднимается отдельный обученный чекпоинт `G`
> загружаемый в одну из трёх архитектур
> (classic `molgan_nets.Generator`, VVRQ `QuantumGenerator`, EFQ
> `QuantumExponentialGenerator`).

```text
app.py
  └─ models/generators.py       реестр моделей и диспетчер генерации
       ├─ models/molgan_nets.py классический графовый генератор
       ├─ models/quantum_model.py  VVRQ / EFQ PQC
       ├─ models/backends.py    исполнение PennyLane / Qiskit
       ├─ models/decode.py      мапинг графа → классический SMILES
       └─ models/molecule_db.py предвалидированные результаты шумных бэкендов
  └─ utils/                     рендер схем и молекул
```

### 2. Локальный запуск приложения

Поддерживается Python **3.11–3.13**. Воспроизводимый способ — через
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python app.py
```

Либо через обычный `pip`:

```bash
python -m pip install -r requirements.txt
python app.py
```

После запуска Gradio поднимет локальный сервер (по умолчанию
`http://127.0.0.1:7860`) 

#### 2.1 Проверки качества

```bash
uv run ruff check app.py models utils scripts tests
uv run pytest -q
```

Тесты покрывают молекулярные метрики, поведение режимов генерации, полноту
упакованных ассетов, безопасность SQL-запросов к SQLite и паритет схем
PennyLane/Qiskit для VVRQ и EFQ.

### 3. Локальный вызов моделей (без UI)

Реестр моделей строится одним вызовом `build_registry`, каждая модель отдаёт
`GenerationResult` со SMILES и метриками:

```python
from models import build_registry

registry = build_registry(dataset="Both")

gen = registry["HQ-MolGAN (VVRQ)"]

# curated: замороженный банк, 100% валидность, воспроизводимо
res = gen.generate(n=8, mode="curated", seed=0)
print(res.smiles)
print(res.validity, res.uniqueness)

# random: живой forward pass на шумовом бэкенде IBM
res = gen.generate(
    n=8,
    mode="random",
    seed=42,
    backend_key="fake_brisbane",   # ideal | fake_brisbane | fake_sherbrooke | fake_osaka
    shots=8192,
)
print(res.smiles, res.validity)
```

Список отображаемых имён — в `models.MODEL_NAMES`; ключи бэкендов — в
`models.BACKEND_KEYS`. `generate(n=...)` где  `n` принимает значения от 1 до 16.

#### 3.1 Пересборка ассетов (опционально)

```bash
# предпосчёт вероятностных векторов PQC на fake-IBM-бэкендах (200k shots)
uv run python scripts/precompute_backends.py

# сборка/расширение SQLite-базы валидированных молекул из банков и probs
uv run python scripts/build_molecule_db.py
```

### 4. Текущие ограничения:

- Квантовые бэкенды реализовыны как локальный (предзаписанные) шумные модели (Fake-IBMs), это **не** живое квантовое железо (которое требует дорого API закрытого в РФ).
- Декодирование случайных графов MolGAN может давать невалидные валентности.
