# Pipeline de RM Cerebral — Documentação Técnica

> Projeto: Segmentação não-supervisionada de ressonâncias magnéticas cerebrais (sequência FLAIR)
> Dataset: `dataset/MR-MS-new` — 24 pacientes (P0001–P0024), formato DICOM

---

## Estrutura de arquivos criados

```
testes/
  janelamento.py              ← código original de referência (janelamento simples)
  visualizador_dicom.py       ← visualizador interativo com janelamento ao vivo
  conversor_janelamento.py    ← conversor em lote: aplica WC/WW e salva novos DICOMs

pipeline/
  segmentacao_pipeline.py     ← pipeline completo de segmentação + visualizador
  cache/                      ← resultados processados (gerado automaticamente)

dataset/
  MR-MS-new/                  ← DICOMs originais (entrada)
  MR-MS-new-janelado/         ← DICOMs com janelamento aplicado (saída do conversor)
```

---

## 1. Visualizador DICOM com Janelamento ao Vivo

**Arquivo:** `testes/visualizador_dicom.py`

**Como executar:**
```bash
python testes/visualizador_dicom.py
```

**O que faz:**
- Carrega toda a série DICOM da pasta configurada em `DICOM_DIR`
- Exibe os slices com janelamento aplicado em tempo real via sliders
- Lê os valores padrão de `WindowCenter` e `WindowWidth` diretamente do header DICOM

**Controles da interface:**

| Controle | Ação |
|---|---|
| Slider **Slice** | Navega entre os cortes axiais |
| Slider **Window Center (WC)** | Desloca o ponto central da janela de visualização |
| Slider **Window Width (WW)** | Aumenta ou reduz o contraste da janela |
| Botão **Resetar Janela** | Volta aos valores originais do header DICOM |
| Teclas `←` `→` `↑` `↓` | Navegação por teclado entre slices |

**Painel de informações (atualizado por slice):**
- ID do paciente, modalidade, dimensões da imagem, pixel spacing
- Mínimo e máximo globais do volume

---

## 2. Conversor DICOM com Janelamento em Lote

**Arquivo:** `testes/conversor_janelamento.py`

**Como executar:**
```bash
python testes/conversor_janelamento.py
```

**O que faz:**
- Percorre todas as 24 pastas de pacientes em `MR-MS-new`
- Para cada arquivo `.dcm`:
  - Lê os pixels originais (aplica `RescaleSlope`/`RescaleIntercept` automaticamente via pydicom)
  - Aplica janelamento com **WC = 1248** e **WW = 2300**
  - Remapeia os valores para `uint16` (0–65535) preservando profundidade máxima de bits
  - Remove `RescaleSlope`, `RescaleIntercept` e `RescaleType` para evitar dupla conversão
  - Força `TransferSyntaxUID = ExplicitVRLittleEndian` (descomprime DICOMs com pixel data comprimido JPEG/JPEG2000)
  - Salva o novo DICOM mantendo todos os demais metadados intactos
- Saída: `dataset/MR-MS-new-janelado/P000X/`

**Parâmetros configuráveis** (topo do arquivo):
```python
WINDOW_CENTER = 1248
WINDOW_WIDTH  = 2300
SOURCE_ROOT   = r"...\dataset\MR-MS-new"
DEST_ROOT     = r"...\dataset\MR-MS-new-janelado"
```

**Fórmula de janelamento:**
```
low    = WC - WW / 2   →   1248 - 1150 = 98
high   = WC + WW / 2   →   1248 + 1150 = 2398
valor  = clip(pixel, low, high)
saída  = (valor - low) / (high - low) × 65535
```

---

## 3. Pipeline de Segmentação Cerebral (FLAIR)

**Arquivo:** `pipeline/segmentacao_pipeline.py`

**Como executar:**
```bash
python pipeline/segmentacao_pipeline.py
```

**Dependências adicionais:**
```bash
pip install scikit-image scikit-learn
pip install SimpleITK   # opcional — habilita N4 Bias Correction e registro afim
```

---

### 3.1 Fluxo do Pipeline

```
DICOM bruto (série 3D)
        │
        ▼
┌─────────────────────────────┐
│  1. Carregamento DICOM       │  pydicom → numpy float32 (Z × H × W)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  2. N4 Bias Field Correction │  SimpleITK (opcional)
│     Otsu mask → N4 filter   │  Itera: [50, 50, 30, 20] por nível
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  3. Skull Stripping          │  Threshold Otsu → fechamento 3D (iter=4)
│     Extração cerebral        │  → fill_holes → maior componente
│                             │  → erosão (iter=3) remove bordas do crânio
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  4. Normalização Z-score     │  μ e σ calculados DENTRO da máscara cerebral
│     Dentro da máscara        │  Voxels externos = 0
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  5. Registro Espacial        │  SimpleITK — transformação afim (opcional)
│     (inter-sequências)       │  Métrica: Mutual Information de Mattes
│                             │  Uso: alinhar T1/T2/FLAIR do mesmo paciente
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  6. Segmentação K-means      │  4 clusters ordenados por intensidade crescente:
│     4 classes (FLAIR)        │  1=CSF, 2=SC (GM), 3=SB (WM), 4=WM hiperintenso
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  7. Detecção de Lesões FLAIR │  Estratégia dupla (A ∪ B):
│     Hemorragias / Placas EM  │  A) Z-score global > 2.0 dentro do cérebro
│                             │  B) Intra-WM: intensidade > WM_mean + 1.5×WM_std
│                             │  → remoção de componentes < 5 voxels (ruído)
└─────────────────────────────┘
        │
        ▼
  Visualizador interativo (tkinter + matplotlib)
```

---

### 3.2 Etapas em detalhe

#### Etapa 2 — N4 Bias Field Correction
Corrige inhomogeneidades do campo magnético que causam variações graduais de intensidade pelo volume (vinheta). Sem essa correção, voxels de mesmo tecido em regiões diferentes do crânio podem ter intensidades distintas, prejudicando a segmentação.

- **Implementação:** `SimpleITK.N4BiasFieldCorrectionImageFilter`
- **Máscara:** Otsu binário para guiar a estimativa do campo de bias
- **Iterações por nível de resolução:** `[50, 50, 30, 20]`
- **Fallback:** se SimpleITK não estiver instalado, esta etapa é pulada sem erro

#### Etapa 3 — Skull Stripping
Remove o crânio, couro cabeludo e ar fora do cérebro para que a segmentação e normalização operem apenas no tecido cerebral.

| Passo | Operação | Parâmetro |
|---|---|---|
| Threshold | Otsu (skimage) | automático |
| Fechamento morfológico | `binary_closing` 3D | iterations=4 |
| Preenchimento | `binary_fill_holes` | — |
| Componente maior | `ndimage.label` | mantém só o maior |
| Erosão | `binary_erosion` | iterations=3 |

#### Etapa 4 — Normalização Z-score
```
z = (x − μ_brain) / σ_brain
```
- `μ` e `σ` calculados apenas sobre os voxels dentro da máscara cerebral
- Voxels externos zerados
- Resultado: distribuição centrada em 0, com desvio padrão 1 — remove viés de escala entre pacientes e equipamentos

#### Etapa 5 — Registro Espacial (função `register_to_reference`)
Disponível quando o paciente possui múltiplas sequências (T1, T2, FLAIR). Alinha geometricamente um volume a outro usando transformação afim.

- **Métrica:** Mutual Information de Mattes (50 bins)
- **Otimizador:** Gradient Descent (lr=1.0, 200 iterações)
- **Inicialização:** centroide geométrico
- **Interpolação:** Linear (bilinear 3D)
- *Não é chamado automaticamente no fluxo principal — invocar manualmente quando necessário*

#### Etapa 6 — Segmentação K-means (4 classes — FLAIR)
Em FLAIR o CSF é suprimido (escuro), tornando-o distinguível da GM. Por isso são usados **4 clusters** em vez de 3.

| Label | Tecido | Cor no viewer |
|---|---|---|
| 0 | Fundo / ar externo | transparente |
| 1 | CSF / Líquor (suprimido) | azul |
| 2 | Substância Cinzenta (GM) | verde |
| 3 | Substância Branca (WM) | amarelo |
| 4 | WM Hiperintenso (lesões FLAIR) | laranja |

Os clusters são ordenados automaticamente por intensidade média crescente após o fit, garantindo correspondência consistente entre pacientes.

#### Etapa 7 — Detecção de Lesões FLAIR (estratégia dupla)

**Detector A — Outlier global (Z-score > 2.0):**
- Seleciona voxels cerebrais com intensidade normalizada acima de 2.0σ
- Captura: edema vasogênico, hemorragias subagudas (isointensas a hiperintensas em FLAIR), grandes lesões

**Detector B — Hiperintensidades intra-WM (WM_mean + 1.5×WM_std):**
- Opera somente sobre voxels classificados como WM (label 3)
- Captura: placas de EM periventriculares, lesões juxtacorticais e infratentoriais que são brilhantes *relativamente à WM normal* mas podem não atingir o threshold global

**Pós-processamento:**
- União A ∪ B
- Remoção de componentes conectados com menos de 5 voxels (ruído pontual)

**Referência clínica para FLAIR:**
- Placas de EM → hiperintensidades periventriculares, juxtacorticais, infratentoriais
- Hemorragia subaguda → isointensa a hiperintensa
- Edema vasogênico → hiperintenso ao redor de lesões expansivas
- Aneurismas → podem aparecer como artefatos de fluxo (hipointensos) ou edema perilesional

---

### 3.3 Parâmetros configuráveis do pipeline

```python
# pipeline/segmentacao_pipeline.py — topo do arquivo

N_TISSUE_CLASSES   = 4      # número de classes K-means (FLAIR = 4)
ANOMALY_Z_THRESH   = 2.0    # Z-score global para detecção de lesões
ANOMALY_WM_STD     = 1.5    # multiplicador de std para detecção intra-WM
ANOMALY_MIN_VOXELS = 5      # tamanho mínimo de componente (voxels)
```

---

### 3.4 Visualizador do Pipeline

Interface desktop (tkinter + matplotlib) com dois painéis lado a lado:

| Painel | Conteúdo |
|---|---|
| Esquerdo | Imagem DICOM original (escala de cinza) |
| Direito | Imagem original + overlays coloridos |

**Controles:**

| Controle | Ação |
|---|---|
| Dropdown **Paciente** | Seleciona qual pasta de paciente processar |
| Botão **▶ Processar Pipeline** | Executa todas as 7 etapas em thread separada |
| Barra de progresso | Mostra progresso em % com descrição da etapa atual |
| Slider **Slice** | Navega entre cortes axiais |
| Slider **Opacidade** | Ajusta transparência dos overlays (0–1) |
| Checkbox **Segmentação de tecidos** | Ativa/desativa overlay de tecidos |
| Checkbox **Lesões / Anomalias** | Ativa/desativa overlay vermelho de lesões |
| Teclas `←` `→` `↑` `↓` | Navegação por teclado |

**Painel Info (por slice):**
- Percentual de cada tecido no slice atual (CSF, GM, WM, WM hiperintenso)
- Contagem de voxels detectados como lesão

**Legenda de cores:**

| Cor | Tecido |
|---|---|
| Azul | CSF / Líquor (suprimido em FLAIR) |
| Verde | Substância Cinzenta |
| Amarelo | Substância Branca |
| Laranja | WM Hiperintenso (FLAIR — 4º cluster K-means) |
| Vermelho | Lesões / Anomalias (detecção dupla) |

---

## 4. Dependências

### Obrigatórias (já no projeto)
```
pydicom >= 2.3.0
numpy
matplotlib
scipy
```

### Adicionais necessárias para os novos scripts
```bash
pip install scikit-learn      # K-means (segmentacao_pipeline.py)
pip install scikit-image      # threshold_otsu (segmentacao_pipeline.py)
```

### Opcional — habilita N4 Bias Correction e Registro Espacial
```bash
pip install SimpleITK
```
> Sem SimpleITK o pipeline executa normalmente, pulando as etapas de N4 e registro com aviso na interface.

---

## 5. Limitações conhecidas e próximos passos

| Limitação | Observação |
|---|---|
| Skull stripping simplificado | Pode incluir resíduos de crânio em imagens com muito sinal de gordura. Para produção considerar **HD-BET** ou **BET (FSL)** |
| K-means sem regularização espacial | Não considera vizinhança espacial. Pode gerar resultados ruidosos em slices com pouco tecido cerebral. Alternativa: **MRF (Markov Random Fields)** ou **GMM-EM** |
| Detecção de lesões por threshold | Sensível a artefatos de movimento e variações de aquisição. Para maior especificidade considerar **LST** (Lesion Segmentation Toolbox) ou modelos supervisionados |
| Aneurismas em FLAIR | Aneurismas aparecem como hipointensidades (flow voids) — o detector atual captura hiperintensidades. Para hipointensidades adicionar threshold `< -2.0σ` |
| Registro espacial não automático | A função `register_to_reference` existe mas precisa ser chamada manualmente quando há múltiplas sequências por paciente |
