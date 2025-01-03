import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kg_base import KG_base
from scipy.stats import skew, kurtosis
from sklearn.neighbors import KNeighborsClassifier as sk_kNN
from sklearn.metrics import accuracy_score as sk_acc
from sklearn.model_selection import train_test_split as sk_tts

class KG_test(KG_base):
    """Génère des jeux de test pour k-guesser.
    On génère des jeux de données et on compare nos variables indépendantes 
    à l'hyperparamètre obtenu de façon classique.
    
    - 'sim()'           pour générer un jeu de 'n' données
    - 'generate()'      pour générer un faux jeu de données
    - 'get_bestk()'     pour déterminer l'hyperparamètre
    - 'get_features()'  pour dériver les 'features'
    - 'save()'          pour sauvegarder les données (append)
    - 'load()'          pour charger les données d'un fichier Excel
    
    Et encore : 
    
    - 'load/save_fds()' pour sauvegarder/charger de faux jeu de données
    - 'show_dataset()'  pour visualiser un faux jeu de données
    - 'show_features()' pour visualiser les 'features' relativement à 'best_k'
    
    Les paramètres pour la génération de données sont des propriétés de la 
    classe. 'head' contient le nom des colonnes pour le jeu de données.
    """
    
    def __init__(self):
        super().__init__()
        # paramètres de génération de faux jeux de données
        self.lim_row = [100, 10000]           # nombre de lignes
        self.lim_ycl = [3, 5]                 # nombre de classes de 'y'
        self.lim_nbx = [3, 9]                 # nombre de 'x'
        self.distr_type = -1                  # type de distribution
        self.distr_noise = -1                 # chevauchement des valeurs

        # Méthodes privées #
        #------------------#
    def _r(self, low, high):
        """Retourne un 'randint' avec numpy, 'high' inclusif.
        (Comportement de la librairie 'random'.)"""
        return np.random.randint(low, high+1)
    def _gen_setup(self):
        """Prépare la génération du jeu de données.
        Le type de distribution et le niveau de bruit sont 'hard-coded'."""
        n_rows = self._r(self.lim_row[0], self.lim_row[1])
        n_y_classes = self._r(self.lim_ycl[0], self.lim_ycl[1])
            # nb of 'x', non-random and distribution type
        cx, ixd = [], 0
        lx = self._r(self.lim_nbx[0], self.lim_nbx[1])
        lxd = self._r(1, lx)                # non-random 'x'
        cx, n_nonrand = [], 0
        for i in range(lx):                 # can produce pure noise
            c = self._r(0, 1)
            if c == 1 and n_nonrand < lxd:  # store distribution type
                cx.append(self.distr_type if self.distr_type > 0 else
                          self._r(1, 3))
                n_nonrand += 1; continue
            cx.append(0)
        del lx; del lxd
        distr_noise = self.distr_noise if self.distr_noise > 0 else \
                      self._r(1, 3)
        return n_rows, n_y_classes, cx, n_nonrand, distr_noise
    def _sim_save(self, f, i, n, dat, tmp, verbose, log_f):
        """Petit bout de code répété dans 'sim()'."""
        self.save(f, tmp)
        if verbose:
            self.log(f"Sauvegardé {i}/{n} dans '{f}'.", log_f, end="\r")
        dat = dat+tmp; tmp = []
        return dat, tmp

        # Charger/sauvegarder les données #
        #---------------------------------#
    def save_fds(self, f, x, y=None):
        """Sauvegarde un faux jeu de données dans un fichier Excel.
        Crée un dossier au besoin."""
        d, fi = os.path.split(f)
        if not os.path.isdir(d):
            os.mkdir(d)
        x, y = self._ifds(x, y)
        columns = [f"x{i}" if i > 0 else "y" for i in range(len(x[0])+1)]
        df = pd.DataFrame(np.insert(x, 0, y, axis=1), columns=columns) \
               .to_excel(f, index=False)
    def load_fds(self, f):
        """Charge un faux jeu de données."""
        if not os.path.exists(f):         # chemin invalide
            return np.array([])
        elif os.path.isdir(f):            # dossier
            return list(filter(None, [pd.read_excel(uf).to_numpy() 
                    if os.path.splitext(uf) == ".xlsx"
                    else None for u in os.listdir(f)]))
        return pd.read_excel(f).to_numpy() # fichier
    def save(self, f, dat):
        """Sauvegarde les données dans un fichier Excel."""
        if not os.path.isfile(f):
            pd.DataFrame(dat, columns=self.head) \
              .to_excel(f, index=False)
        else:
            df = pd.read_excel(f)
            nr = df[df.columns[0]].count()
            for i, row in enumerate(dat):
                df.loc[nr+i] = row
            df.to_excel(f, index=False)
    def load(self, f):
        """Charge les données d'un fichier Excel."""
        if not os.path.isfile(f):
            return np.array([])
        dat = pd.read_excel(f)
        return pd.read_excel(f).to_numpy()

        # Visualiser #
        #------------#
    def show_dataset(self, x, y=None, cx=None):
        """Présente le jeu de données comme nuages de points.
        'cx' est une liste indiquant quels 'x' sont aléatoires (alpha)."""
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        x, y = self._ifds(x, y)
        n_rows, n_cols = x.shape
        lx = np.arange(1, n_rows+1, 1) # x-axis (row-index)
        for i in range(0, n_cols):     # y-axis (x-value)
            alpha = 0.7 if not cx else 0.4 if cx[i] == 0 else 0.8
            ax[0].scatter(lx, x[:,i], alpha=alpha)
            alpha = 0.7 if not cx else 0.1 if cx[i] == 0 else 0.8
            ax[1].scatter(x[:,i], y, alpha=alpha)
        ax[0].set_xlabel("Row index")
        ax[0].set_ylabel("X-value")
        ax[1].set_xlabel("X-value")
        ax[1].set_ylabel("Y-value")
        fig.tight_layout()
        plt.show()
    def show_features(self, dat, n_col=3):
        """Présente la relation de chaque paramètre avec 'y'.
        'n_col' contrôle le nombre de colonnes affiché."""
        if dat.shape[0] == 0:
            return
        x, y = self._ifds(dat)
        lx = np.arange(1, x.shape[0], 1)
        r = (x.shape[1]//n_col); r = r+1 if x.shape[1]%n_col > 0 else r
        c = n_col if x.shape[1] >= n_col else x.shape[1]
        fig, ax = plt.subplots(r, c, figsize=(12, 6))
        for i in range(x.shape[1]):
            r = int(np.floor(i/n_col)); c = i-(n_col*r)
            ax[r,c].scatter(x[:,i], y)
            ax[r,c].set_title(f"{self.head[i+1]}")
        c += 1
        for j in range(c, n_col):
            fig.delaxes(ax[r, j])
        fig.supxlabel("Feature values")
        fig.supylabel("best_k")
        fig.suptitle("Feature-to-best_k relation", fontsize=18)
        fig.tight_layout()
        fig.show()
    
        # Générer #
        #---------#
    def generate(self, view=False, cheat=True):
        """Génère un faux jeu de données."""
        n_rows, n_y_classes, cx, n_nonrand, distr_noise = self._gen_setup()
        gy, gx = [], []
        for r in range(n_rows):
            y = self._r(0, n_y_classes-1)   # valeur de 'y'
            yr = (y+1)/(n_y_classes+1)      # ratio de non-aléatoires
            gy.append([y]); gx.append([])
            for c in cx:
                match c: # choix de distribution...
                    case 0: # ... aléatoire
                        x = self._r(0, 1000)
                    case 1: # ... linéaire
                        x = self._r((1000*yr)-(100*distr_noise),
                                    (1000*yr)+(100*distr_noise))
                    case 2: # ... quadratique
                        x = self._r((1000*(yr**2))-(100*distr_noise),
                                    (1000*(yr**2))+(100*distr_noise))
                    case 3: # ... partielle (linéaire)
                        x = self._r(0, 1000) if self._r(0, 100) > 90 else \
                            self._r((1000*yr)-(100*distr_noise),
                                    (1000*yr)+(100*distr_noise))
                x = 0 if x < 0 else 1000 if x > 1000 else x
                gx[-1].append(x)
        gx, gy = np.array(gx), np.array(gy).ravel()
        if view:
            self.show_dataset(gx, gy, cx)
        if cheat:
            d_vi = {
                'n_nonrand': n_nonrand/len(cx),
                'distr_typ': sum(cx)/n_nonrand if n_nonrand > 0 else 0,
                'distr_noise': distr_noise
            } # cheating to provide values for our model
            return gx, gy, d_vi
        else:
            return gx, gy
    def get_bestk(self, x, y=None):
        """Teste l'hyperparamètre 'k' (nombre de voisins)
        pour le jeu de données 'ds'.
        L'intervalle est limité à [1, sqrt(len(x))]."""
        x, y = self._ifds(x, y)               # check for dataset
        x_tr, x_te, y_tr, y_te = sk_tts(x, y) # train/test
        rk, ra, ly = 0, -1., int(np.sqrt(len(y)))-1
        for k in np.arange(1, ly, 1):         # all possible k's
            m = sk_kNN(k)                     # kNN model
            m.fit(x_tr, y_tr)                 # fit on training data
            yp_te = m.predict(x_te)           # predict on test data
            a = sk_acc(y_te, yp_te)           # get accuracy score
            if a > ra:                        # pick best 'k'
                ra, rk = a, k
        return rk
    def get_features(self, x, y=None, d_vi=None):
        """Dérive des variables (pour déterminer l'hyperparamètre 'k')
        à partir d'un jeu de données (x, y).
        Voir 'self.head' pour comrpendre le contenu de la liste retournée.
        'd_vi' est un moyen de tricher pour obtenir des variables 
        normalement inaccessibles."""
        x, y = self._ifds(x, y)                # check for dataset
        n_row, n_feat = x.shape; n_mult = n_row*n_feat
        if n_row <= 0:                         # 0 datapoint, no point
            return [-1. for i in range(1, len(self.header))]
        l_vi = [n_row,                         # n_rows
                len(np.unique(y)),             # n_y_classes
                n_feat                         # n_features
        ] # list to return
        var, skw, krt = 0., 0., 0.             # pre-calculate more data
        for i in range(0, n_feat):
            var += np.var(x[:,i]); skw += skew(x[:,i])
            krt += kurtosis(x[:,i])
        var, skw, krt = var/n_mult, skw/n_mult, krt/n_mult; del n_mult
        for k in self.head[4:7]:               # check 'd_vi' for some values
            if k in d_vi:                      # 'd_vi' handles it
                l_vi.append(d_vi[k]); continue
            match k:
                case 'n_nonrand':              # feature selection
                    l_vi.append(np.sum(self.select_features(x, y)))
                case 'distr_typ':              # random if no cheating
                    l_vi.append(np.random.uniform(1, 3))
                case 'distr_noise':            # random if no cheating
                    l_vi.append(self._r(1, 3))
                case _:
                    continue
        l_vi = l_vi + [var, skw, krt]
        return l_vi
    
        # Méthode principale #
        #--------------------#
    def sim(self, n=100, f="", verbose=False, log_f=""):
        """Génère 'n' données."""
        dat, tmp = [], []
        for i in range(n):
            x, y, d_vi = self.generate()                    # fake dataset
            tmp.append([self.get_bestk(x, y)]+
                        self.get_features(x, y, d_vi))      # datapoint
            if verbose:
                self.log(f"Point de donnée {i} généré...", log_f, end="\r")
            if f and i > 0 and i%20 == 0:                   # save every 20
                dat, tmp = self._sim_save(f, i, n, dat, tmp, verbose, log_f)
        if f:
            dat, tmp = self._sim_save(f, n, n, dat, tmp, verbose, log_f)
        return np.array(dat+tmp)

