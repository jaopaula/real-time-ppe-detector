# 🦺 Real-Time PPE Detector — YOLO9 / YOLO8
### Detecção em tempo real de EPIs (Capacete + Óculos)

Este repositório implementa detecção de EPIs em tempo real usando modelos YOLO9-E e YOLO8, com suporte completo a GPU NVIDIA + CUDA, detecção via webcam e possibilidade de treinar modelos customizados.

---

# 🚀 Pipeline Completa (GPU → CUDA → Torch → YOLO → Webcam)

Este README documenta exatamente a pipeline real que você utilizou, incluindo verificações necessárias.

---

# 1️⃣ Criar ambiente virtual

```
python -m venv .venv
```

---

# 2️⃣ Ativar o ambiente

Windows:
```
.venv\\Scripts\\activate
```

---

# 3️⃣ Instalar dependências do projeto

```
pip install -r requirements.txt
```

---

# 4️⃣ Instalar PyTorch GPU (fundamental)

Compatível com a GTX 1660 SUPER:

```
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

# 5️⃣ Verificar GPU e CUDA

### 5.1 Torch detectou a GPU?
```
python -c "import torch; print(torch.cuda.is_available())"
```

### 5.2 Ver driver e GPU instalada
```
wmic path win32_VideoController get name,driverversion
```

### 5.3 Teste completo (script do projeto)
```
python scripts/check_cuda.py
```

Saída esperada:
```
Torch version: 2.5.1+cu121
CUDA disponível: True
Nome da GPU: NVIDIA GeForce GTX 1660 SUPER
```

---

# 6️⃣ Baixar o modelo YOLO

Modelo recomendado: YOLO9-E

Coloque o arquivo na pasta:
```
models/yolo9e.pt
```

---

# 7️⃣ Dataset (opcional — para treino)

Dataset oficial utilizado:
🔗 https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection

Estrutura esperada em caso de treino:
```
data/
 ├── train/
 ├── valid/
 ├── test/
 └── data.yaml
```

---

# 8️⃣ Detecção em tempo real via webcam

Uso básico:
```
python detect_webcam.py --weights models/yolo9e.pt --device 0
```

Parâmetros úteis:
```
--device 0     # GPU
--conf 0.20    # confiança mínima
--imgsz 960    # maior precisão
```

Exemplo completo:
```
python detect_webcam.py --weights models/yolo9e.pt --device 0 --conf 0.20 --imgsz 960
```

---

# 9️⃣ Treinar seu próprio modelo (opcional)

```
python train.py
```

Saídas ficam em:
```
runs/detect/
```

---

# 🔟 Estrutura Completa do Projeto

```
real-time-ppe-detector/
│── data/
│── models/
│   └── yolo9e.pt
│── runs/
│── scripts/
│   └── check_cuda.py
│── detect_webcam.py
│── train.py
│── requirements.txt
└── README.md
```

---

# 🧹 .gitignore recomendado

```
__pycache__/
*.py[cod]
.venv/
.vscode/
.idea/
runs/
models/*.pt
data/
.DS_Store
Thumbs.db
```

---

# 💡 Dicas pro TCC

✔ YOLO9-E = melhor precisão
✔ imgsz 960 melhora óculos/capacete
✔ use device 0
✔ não commit modelos pesados
✔ mantenha scripts minimalistas

---

# 👨‍💻 Autor
Projeto configurado para o TCC de João — USCS.
