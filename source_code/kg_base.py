import os
import numpy as np
import pandas as pd

class KG_base():
    """Classe "abstraite" pour k-guesser.
    Contient des méthodes communes.
    
    - 'log()'             permet de journaliser les opérations,
                          en console ou dans un fichier.
    - 'params()'          permet d'obtenir et de modifier les 
                          propriétés (publiques) de l'instance.
    - 'select_features()' filtre les variables par méthode de 
                          corrélation.
    """
    def __init__(self):
        self.log_path = ""
        self.head = ['best_k', 'n_rows', 'n_classes', 'n_features',
                     'n_nonrand', 'distr_type', 'distr_noise', 
                     'mean_var', 'mean_skew', 'mean_kurtosis']

    def _ifds(self, x, y=None):
        """Sépare 'x/y' si ce n'est pas déjà fait."""
        if y is not None:                            # déjà séparé
            return x, y
        elif isinstance(x, np.ndarray):              # numpy array
            return x[:,1:], x[:,0]
        elif isinstance(x, pd.core.frame.DataFrame): # pandas DataFrame
            yn = x.columns[0]
            return x.drop([yn]), x[yn]
        else:                                        # (supposé) list
            y = []
            for i,ix in x:
                y = x[i][0]; x[i] = x[i][1:]
            return np.array(x), np.array(y)

    def log(self, txt, f="", end="\n"):
        """Journalise les opérations.
        Permet d'écrire dans un fichier ou au moins évite les 'print'
        dans le code, généralement utilisés pour débugger."""
        f = self.log_path if not f else f
        if f:
            mode = "w" if not os.path.isfile(f) else "a"
            with open(f, mode=mode, encoding="utf-8") as wf:
                wf.write(txt+end)
        else:
            print(txt, end=end)

    def params(self, d_params={}):
        """Fournit les paramètres de la classe.
        Permet aussi de changer ces paramètres."""
        d_cpy = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):     # pas de propriété privée
                continue
            d_cpy[k] = v
        for k, v in d_params.items(): # édition de propriété
            if k in d_cpy:
                d_cpy[k] = self.__dict__[k] = v
        return d_cpy                  # copie par sécurité
      
    def select_features(self, x, y=None, tol=0.2, delete=False):
        """Utilise la corrélation pour éliminer les catégories de 'x'
        considérées non-significatives.
        Retourne la liste d'index de variables corrélées ; si 
        'delete=True', retourne 'x' sans ces colonnes."""
        x, y = self._ifds(x, y)
        l_index = []
        for i in range(x.shape[1]): # correl' variable par variable
            ix = x[:,i] if isinstance(x, np.ndarray) else \
                 pd.DataFrame(x.iloc[:,i].values.reshape(-1, 1))
            corr = np.corrcoef(y, ix)[0][1]
            corr = corr*-1 if corr < 0 else corr   # valeur absolue
            if corr > tol:
                l_index.append(i)
        if delete:                                 # retirer de 'x'
            l_rem = list(filter(None, [i if i not in l_index else None 
                                       for i in range(x.shape[1])]))
            if isinstance(x, pd.core.frame.DataFrame):
                x.drop(x.columns[l_rem], axis=1, inplace=True)
            elif isinstance(x, np.ndarray):
                x = np.delete(x, l_rem, 1)
            return x
        return l_index