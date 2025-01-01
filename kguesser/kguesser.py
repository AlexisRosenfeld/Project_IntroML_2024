import os, time, re, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn.neighbors import KNeighborsClassifier as sk_kNN
from sklearn.metrics import accuracy_score as sk_acc
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.metrics import r2_score as sk_r2
from sklearn.model_selection import train_test_split as sk_tts
from sklearn.model_selection import cross_val_score as sk_cross
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression as sk_lm
from sklearn.neural_network import MLPRegressor as sk_nn

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

        # Charger/sauvegarder #
        #---------------------#
    def save(self, f, dat, columns=None):
        """Sauvegarde les données dans un fichier Excel."""
        columns = self.head if columns is None else columns
        if not os.path.isfile(f):
            pd.DataFrame(dat, columns=columns) \
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

    def show(self, x, y=None, f="", columns=[]):
        """Imprime la tête des données."""
        x, y = self._ifds(x, y)
        columns = self.head if not columns else columns
        if isinstance(y, np.ndarray):
            df = pd.DataFrame(np.hstack((y.reshape(-1, 1),x)), columns=columns)
        else:
            df = pd.concat([y, x], axis=1, join="inner")
        self.log(df.head(), f)
    
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
    def _sim_save(self, f, i, n, dat, tmp, verbose, log_f, columns=None):
        """Petit bout de code répété dans 'sim()'."""
        self.save(f, tmp, columns)
        if verbose:
            self.log(f"Sauvegardé {i}/{n} dans '{f}'.", log_f, end="\r")
        dat = dat+tmp; tmp = []
        return dat, tmp

        # Charger/sauvegarder #
        #---------------------#
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
    def generate(self, view=False, cheat=True, save_path=""):
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
        if save_path:
            self.save_fds(save_path, gx, gy)
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
        rk, ra, ly = 0, -1., 101 # int(np.sqrt(len(y)))-1
        l_bestk = np.array([0. for i in range(ly+1)])
        for k in np.arange(1, ly, 1):         # all possible k's
            try:
                m = sk_kNN(k)                     # kNN model
                m.fit(x_tr, y_tr)                 # fit on training data
                yp_te = m.predict(x_te)           # predict on test data
                a = sk_acc(y_te, yp_te)           # get accuracy score
                if a > ra:                        # pick best 'k'
                    l_bestk[0], l_bestk[1] = ra, rk = a, k
                l_bestk[k+1] = a
            except:
                break
        return rk, l_bestk
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
                len(np.unique(y)),             # n_classes
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
                case 'distr_type':              # random if no cheating
                    l_vi.append(np.random.uniform(1, 3))
                case 'distr_noise':            # random if no cheating
                    l_vi.append(self._r(1, 3))
                case _:
                    continue
        l_vi = l_vi + [var, skw, krt]
        return l_vi
    
        # Méthode principale #
        #--------------------#
    def sim(self, n=100, df="data.xlsx", kf="k.xlsx", verbose=False, log_f=""):
        """Génère 'n' données."""
        d, dt, k, kt = [], [], [], []
        cols = ["best_k", "bes_acc"]+[str(i+1) for i in range(100)]
        for i in range(n):
            x, y, d_vi = self.generate()                    # fake dataset
            nk, lk = self.get_bestk(x, y)
            dt.append([nk]+self.get_features(x, y, d_vi))   # datapoint
            kt.append(lk)
            if verbose:
                self.log(f"Point de donnée {i} généré...", log_f, end="\r")
            if df and i > 0 and i%20 == 0:                   # save every 20
                k, kt = self._sim_save(kf, i, n, k, kt, False, "", cols)
                d, dt = self._sim_save(df, i, n, d, dt, verbose, log_f)
        if df:
            k, kt = self._sim_save(kf, n, n, k, kt, False, "", cols)
            d, dt = self._sim_save(df, n, n, d, dt, verbose, log_f)
        return np.array(d+dt), np.array(k+kt)

class KG_model(KG_base):
    """Génère, entraîne et teste le modèle.
    Pour le moment, ne gère que la régression linéaire.
    
    - 'fit/predict()' reprennent la librairie scikit-learn.
    - 'test()' teste la précision du modèle.
    - 'preprocess()' gère les variables catégorielles et le 'scaling'.
    - 'select()' fait de la 'feature selection'.
    
    Dans le futur, la classe devrait aussi gérer d'autres modèles 
    quantitatifs pour le projet k-guesser, et permettre de continuer
    l'entraînement d'un modèle existant via 'river' (?)."""
    
    def __init__(self, m_path="", k_path=""):
        super().__init__()
        self.m = None
        self.m_path = "regression_model.pkl" if not m_path else m_path
        self.ktab = self.load("k.xlsx" if not k_path else k_path) # table
    
        # Méthodes privées #
        #------------------#   
    def _stats(self, x, y, yp, params):
        """Fournit la déviation standard et les t/p-valeurs pour 'fit()'."""
        g = pd.DataFrame({'Constant':np.ones(x.shape[0])}) \
              .join(x.reset_index(drop=True))
        dd = (g.shape[0]-g.shape[1])
        u_hat = y.values-yp.values
        sigma_squared_hat = (u_hat.T@u_hat)[0,0] / dd
        st_std = (np.linalg.inv(g.T@g)*sigma_squared_hat).diagonal()
        print("std:",st_std)
        st_tval = params/np.sqrt(st_std)
        st_pval = 2*(1-stats.t.cdf(np.abs(st_tval), (g.shape[0]-g.shape[1])))
        ouptut = pd.DataFrame({
            'Coefficients': params,
            'Standard Errors': st_std,
            't-values': st_tval,
            'p-values': st_pval
        })
        return output
    def _test_prep(self, x, y, preprocess, verbose):
        """Test d'un modèle : pré-traitement."""
        x, y = self._ifds(x, y)
        if preprocess:                            # pré-traitement
            x, y = self.preprocess(x, y, verbose=False)
        # x_tr, x_val, y_tr, y_val = sk_tts(x, y)   # entraînement/validation
        x_tr, x_val, y_tr, y_val, _, li = self.splitmerge(x, y)
        return x_tr, x_val, y_tr, y_val, li
    def _test_run(self, x_tr, x_val, y_tr, y_val, cross, verbose, li):
        """Test d'un modèle : évaluation."""
        if verbose:
            self.log("Starting cross-validation...              ", end="\r")
        if cross:                                 # validation croisée
            sc = sk_cross(self.m, x_tr, y_tr, cv=5) 
            mean, std = sc.mean(), sc.std()
        else:
            mean, std = -1., -1.
        if verbose:
            self.log("Starting evalution...                     ", end="\r")
        self.fit(x_tr, y_tr, verbose=False)       # entraînement général
        mse, r2 = self.eval(x_val, y_val, verbose, li) # évaluation générale
        if verbose:
            self.log(
                "#----------------------------------------\n"+
                "|Model Performance:\n"+
                f"|Cross-validation mean: {mean:.02f} "
                f", std: {std:.02f}\n"+
                f"|Mean Square Error: {mse:.02f}\n"+
                f"|R² Score: {r2:.02f}\n"+
                "#----------------------------------------\n"
            )
        return mean, std, mse, r2
        # Sauvegarde/chargement #
        #-----------------------#
    def save_m(self, f):
        """Sauvegarde le modèle via joblib."""
        joblib.dump(self.m, f)
    def load_m(self, f):
        """Récupère le modèle sauvegardé via joblib.
        Attention, c'est une faille de sécurité."""
        f = self.m_path if not f else f
        self.m = joblib.load(f) if os.path.isfile(f) else self.m
    def load_k(self, f="k.xlsx"):
        """Charge la table k->précision (voir 'KG_test.best_k()'."""
        self.ktab = self.load(f)

        # Pré-traitement #
        #----------------#
    def preprocess(self, x, y=None, columns=['distr_noise'],
                         select=True, verbose=True):
        """Variables catégorielles et équilibrage.
        Note : on assume que le jeu de données est propre."""
        x, y = self._ifds(x, y)
            # Catégorisation (OneHot)
        do = [True if self.head[i] in columns else False 
              for i in range(1, len(self.head))]
        encoder, names = OneHotEncoder(sparse_output=False), ['best_k']
        nx = None
        for i in range(x.shape[1]):
            n, ix = self.head[i+1], np.reshape(x[:,i], (-1, 1))
            if not do[i]:
                nx = np.append(nx, ix, 1) if nx is not None else ix
                names.append(n); continue
            ix = encoder.fit_transform(ix)
            nx = np.append(nx, ix, 1) if nx is not None else ix
            for j in range(ix.shape[1]):
                names.append(n+"_"+str(j+1))
        x = nx; del nx
        if verbose:
            self.log(f"OneHotEncoded:\n\t {names}")
            # Équilibrage
        x = StandardScaler().fit_transform(x) # scaling
        for i in range(x.shape[1]):           # fix categoricals
            xu = np.unique(x[:,i])
            if len(xu) > 2:
                continue
            for j in range(len(x[:,i])):
                v = x[j,i]
                nv = xu[1] if v == xu[0] else xu[0]
                x[j,i] = 0 if nv > v else 1
        if verbose:
            self.log("Scaled...                              ", end="\r")
            # Sélection de variables
        if select:
            l_i = self.select_features(x, y)           # par corrélation
            l_cat = [names[n+1] for n in l_i 
                     if re.search(r"_\d$", names[n+1])]# conserver catégories
            for i, n in enumerate(names):
                if not re.search(r"_\d$", n):
                    continue
                elif not i-1 in l_i:
                    l_i.append(i-1)
            l_rem = np.arange(0, x.shape[1], 1)
            l_rem = [i for i in l_rem if i not in l_i]
            l_rem.reverse()
            x = np.delete(x, l_rem, 1) # not DataFrame compatible
            for i in l_rem:
                names.pop(i+1)
            if verbose:
                self.log(f"Selected:\n\t {names}")
            # Passage au DataFrame ?
        # x = pd.DataFrame(x, columns=names[1:])
        # y = pd.DataFrame(y, columns=[names[0]])
        return x, y
    def splitmerge(self, x, y=None, ratio=0.8):
        """Une implémentation manuelle de scikit-learn 'train_test_split'
        pour garder un index 'li' de chaque donnée."""
        li, l1, l2 = [i for i in range(x.shape[0])], [], []
        x1, x2, y1, y2 = [], [], [], []
        n = int(x.shape[0]*ratio)
        if n <= 0:                            # I don't know what went wrong
            return x, x2, y, y2, li, l2
        c, l_c = 0, np.random.choice(li, n, replace=False)
        l_c.sort()
        for i in li:
            if l_c[c] == i:
                x1.append(x[i,:]); y1.append(y[i]); l1.append(i); c += 1
            else:
                x2.append(x[i,:]); y2.append(y[i]); l2.append(i)
        return np.array(x1), np.array(x2), np.array(y1), np.array(y2), l1, l2
    
        # Fit/predict #
        #-------------#
    def fit(self, x, y, verbose=False):
        """Entraîne le modèle.
        Note : on ne peut pas encore 'continuer' l'entraînement du modèle."""
        self.m.fit(x, y)                      # entraînement
        if verbose:
            # yp = pd.DataFrame({'best_k':self.m.predict(x).ravel()})
            # params = np.append(self.m.intercept_, self.m.coef_)
            # df_print = self._stats(x, y, yp, params)
            # df_print.index = ['Intercept']+list(x.columns)
            # self.log(df_print)
            txt = (f"Fit:\n\tIntercept: {self.m.intercept_[0]:.02f}\n"+
                   "Coefficients:\n")
            for i, colname in enumerate(x.columns):
                txt = txt+f"\t{colname}: {self.m.coef_[0][i]:.02f}\n"
            self.log(txt)
    def predict(self, x):
        """Retourne le 'best_k'."""
        return self.m.predict(x)              # prédiction
    def eval(self, x, y, verbose=True, li=[]):
        """Évalue le modèle déjà entraîné, sans validation croisée."""
        yp = self.predict(x)                   # évaluation
        if li and self.ktab.shape[0] > 0:
            yp = [self.ktab[li[i], int(np.round(yi))] 
                  for i, yi in enumerate(yp)]
            y = [self.ktab[li[i], 1] for i in range(len(y))]
        return sk_mse(y, yp), sk_r2(y, yp) # MSE et R2

        # Méthode principale #
        #--------------------#
    def test_lm(self, x, y=None, params={}, preprocess=True, 
                cross=True, verbose=True):
        """Méthode générale pour tester le modèle 
        à partir d'un jeu de données."""
        x_tr, x_val, y_tr, y_val, li = self._test_prep(x, y, 
                                                       preprocess, verbose)
        self.m = sk_lm()
        return self._test_run(x_tr, x_val, y_tr, y_val, cross, verbose, li)
    def test_nn(self, x, y=None, params={}, preprocess=True, 
                cross=True, verbose=True):
        """Méthode générale pour tester un réseau neuronal 
        à partir d'un jeu de données."""
        x_tr, x_val, y_tr, y_val, li = self._test_prep(x, y, 
                                                   preprocess, verbose)
        pm = {
            'hidden_layer_sizes':[x_tr.shape[1], 64, 64, 32],
            'activation':'relu',
            'solver':'lbfgs',
            'max_iter':10000,
            'early_stopping':True,
            'verbose':False
        }
        for k,v in params.items():
            pm[k] = v
        self.m = sk_nn(**pm)
        if isinstance(x, pd.core.frame.DataFrame):
            y_tr, y_val = y_tr.to_numpy().ravel(), y_val.to_numpy().ravel()
        return self._test_run(x_tr, x_val, y_tr, y_val, cross, verbose, li)
        
if __name__ == "__main__":
    kg_test = KG_test()
    dat, k = kg_test.sim(100, "", True, "")
    # dat = kg_test.load("test.xlsx")
    # kg_model = KG_model()
    # kg_model.load_k("k.xlsx")
    # mean, std, mse, r2 = kg_model.test_lm(dat)
    # mean, std, mse, r2 = kg_model.test_nn(dat, cross=False)
    print("Done.")