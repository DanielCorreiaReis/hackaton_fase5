# 🛡️ STRIDE OCR Classifier and Report Generator

Este projeto realiza classificação automática de diagramas de arquitetura a partir de uma imagem, identifica componentes utilizando OCR da Azure Cognitive Services e gera um relatório STRIDE em PDF com formatação avançada.

---

## 📂 Estrutura do Projeto

- **treina_modelo.py**  
  Script para treinar e salvar um modelo de classificação de arquiteturas com Naive Bayes e `CountVectorizer`.

- **mvp_ocr_classificador_stride_pdf.py**  
  Script principal que:
  - Extrai texto do diagrama via Azure OCR.
  - Classifica o tipo de arquitetura.
  - Realiza correspondência inteligente com um modelo STRIDE (JSON).
  - Gera relatório em PDF.

---

## ⚙️ Requisitos

### Instale as dependências com:

```
pandas
scikit-learn
joblib
requests
python-dotenv
reportlab
azure-identity
azure-keyvault-secrets
```

---

## 🚀 Como Usar

### 1️⃣ Treinar o Modelo

Para criar o classificador:

```bash
python treina_modelo.py
```

Este script irá gerar:

- `modelo_treinado.pkl`
- `vectorizer.pkl`

> O arquivo `dataset_componentes.csv` deve existir com pelo menos duas colunas:
> - `componentes`: texto dos componentes
> - `tipo`: rótulo da arquitetura

---

### 2️⃣ Executar OCR, Classificação e Gerar Relatório

Edite o caminho da imagem no script (`IMAGE_PATH`) e rode:

```bash
python mvp_ocr_classificador_stride_pdf.py
```

Ao finalizar, será criado um PDF chamado `relatorio_stride.pdf` contendo:

- O tipo de arquitetura detectado.
- Componentes reconhecidos.
- Ameaças e contramedidas do modelo STRIDE.

---

## 📄 Exemplo de Uso

```bash
python treina_modelo.py
python mvp_ocr_classificador_stride_pdf.py
```

Saída esperada:

```
Fonte registrada: SegoeUI
Extraindo texto do diagrama...
   → Componentes detectados: ['web_server', 'sql_database']
Classificando arquitetura...
   → Arquitetura prevista: web_app
Filtrando STRIDE...
Gerando PDF...
✅ Relatório salvo: relatorio_stride.pdf
```

---

## ✨ Funcionalidades

- Reconhecimento óptico de caracteres (OCR) com Azure Cognitive Services.
- Classificação automática via Naive Bayes.
- Correspondência inteligente de tokens (Jaccard e SequenceMatcher).
- Geração de PDF Unicode com fonte personalizada.
- Compatível com Windows e Linux.
