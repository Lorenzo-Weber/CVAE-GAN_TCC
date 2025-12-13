import numpy as np
import matplotlib.pyplot as plt

class Filter():
    def __init__(self, split=0.04):

        """
        Filtra os espectros gerados pela CVAE-GAN

        Atributos:
            split (float): porcentagem de espectros a serem mantidos --> pega os top split%  

        O que ja foi testado (evitar repeticao):
            - Filtro usando distancia euclidiana de todos espectros (atual, funciona bem, parece que tem abordagens melhores mas essa foi a que se saiu melhor)
            - Filtro usando distancia euclidiana centrada em 0, subtrair o primeiro valor do espectro de todo o espectro, dando um shift para 0 (pior que o atual)            
            - Filtro usando similaridade de cosseno (pior que o atual)
            - Filtro misturando distancia euclidiana e similaridade de cosseno (pior que o atual)
        """

        self.split = split

    def filter(self, fakes, y_fakes, reals):

        """
            Compara os espectros gerados com os reais e seleciona os mais proximos
            Utiliza a distancia euclidiana para medir a proximidade (e vai acumulando a soma)

            Atributos:
                fakes (array): espectros gerados pela CVAE-GAN
                y_fakes (array): labels dos espectros gerados
                reals (array): espectros reais para comparar
        """

        total_dists = []
        for fake in fakes:
            dists = np.linalg.norm(reals - fake, axis=1)
            total_dists.append(np.sum(np.abs(dists)))

        total_dists = np.array(total_dists)

        n_select = int(len(fakes) * self.split)
        top_indices = np.argsort(total_dists)[:n_select]

        selected_fakes = fakes[top_indices]
        selected_y = y_fakes[top_indices]

        print(f'Sobraram {len(selected_fakes)} amostras apos o filtro')

        return selected_fakes, selected_y
