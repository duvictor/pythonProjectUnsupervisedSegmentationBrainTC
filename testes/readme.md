O que o visualizador faz:
•
Carrega todos os 284 arquivos .dcm da pasta P0008
•
Exibe a imagem com janelamento aplicado em tempo real
•
Sliders:
◦
Slice — navega entre os cortes
◦
Window Center (WC) — ajusta o centro da janela
◦
Window Width (WW) — ajusta a largura da janela
•
Teclado: setas ←/→ (ou ↑/↓) para navegar entre slices
•
Botão "Resetar Janela" — volta aos valores originais do DICOM
•
Painel de info com ID do paciente, modalidade, tamanho e pixel spacing
Usa apenas tkinter (nativo do Python) + matplotlib + pydicom + numpy.