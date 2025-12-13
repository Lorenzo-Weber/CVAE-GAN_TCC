# CVAE-GAN para Geração de Espectros NIR

Este repositório implementa uma **CVAE-GAN** (Conditional Variational Autoencoder + Generative Adversarial Network) utilizada no meu trabalho de conclusao de curso.

O modelo foi pensado para trabalhar com espectros unidimensionais (vetores) e rótulos condicionais, sendo adequado para tarefas de aumento de dados (*data augmentation*) em problemas de classificação ou regressão.

---

## Base de uma GAN

Uma **GAN** é composta por dois modelos que competem entre si:

* **Gerador (G)**: aprende a gerar dados sintéticos que se pareçam com os dados reais.
* **Discriminador (D)**: aprende a distinguir dados reais de dados gerados.

Durante o treinamento:

1. O gerador cria amostras falsas.
2. O discriminador avalia se as amostras são reais ou falsas.
3. O gerador é penalizado quando o discriminador identifica suas amostras como falsas.
4. O discriminador é penalizado quando erra a classificação.

Esse processo adversarial força o gerador a produzir dados cada vez mais realistas.

---

## Extensão para CVAE-GAN

Neste projeto, a GAN é combinada com um **VAE condicional**:

* O **Encoder** aprende uma representação latente (média e variância) dos espectros reais.
* O **Decoder / Gerador** reconstrói ou gera novos espectros a partir do espaço latente.
* A geração é **condicional**, ou seja, utiliza rótulos (classes ou atributos) como entrada adicional.

Com isso, o modelo aprende:

* Estrutura latente dos espectros (VAE)
* Realismo estatístico e visual (GAN)

---

## Pré-processamentos espectrais

Antes do treinamento, os espectros passam por transformações clássicas em espectroscopia NIR.

### MSC – *Multiplicative Scatter Correction*

O **MSC** corrige variações multiplicativas e aditivas causadas por espalhamento da luz, diferenças de caminho óptico ou granulometria da amostra.

Intuição:

* Ajusta cada espectro em relação a um espectro de referência (geralmente a média).
* Reduz efeitos não químicos, preservando a informação espectral relevante.

---

### SNV – *Standard Normal Variate*

O **SNV** normaliza cada espectro individualmente:

* Subtrai a média do espectro.
* Divide pelo desvio padrão.

Isso reduz efeitos de espalhamento e escala, tornando os espectros mais comparáveis entre si.

---

## Filtro aplicado aos espectros gerados

Após a geração, é aplicado um **filtro nos espectros sintéticos** para remover amostras fisicamente incoerentes ou ruidosas.

Esse filtro basicamente calcula a distancia euclidiana entre o espectro sintetico gerado com todos os reais e seleciona os com a menor distancia

O objetivo é garantir que apenas espectros plausíveis sejam adicionados ao conjunto de dados.

---

## Penalização pela primeira derivada

Além das perdas tradicionais da GAN e do VAE, o treinamento inclui uma **penalização baseada na primeira derivada do espectro**.

Ideia central:

* Espectros NIR reais tendem a ser **suaves**.
* Grandes variações ponto a ponto indicam ruído ou artefatos.

Implementação conceitual:

* Calcula-se a primeira derivada do espectro real e do gerado.
* Penaliza-se a diferença entre essas derivadas.

Efeito:

* Reduz oscilações artificiais.
* Força o gerador a respeitar a continuidade espectral.

---

## Resumo do funcionamento

1. Os espectros reais são pré-processados (MSC / SNV).
2. O Encoder aprende a distribuição latente.
3. O Gerador cria espectros condicionados aos rótulos.
4. O Discriminador avalia real vs. sintético.
5. O treinamento considera:

   * Loss adversarial (GAN)
   * Loss de reconstrução (VAE)
   * KL divergence
   * Penalização pela primeira derivada
6. Os espectros gerados passam por um filtro final antes do uso.

