import os, re, joblib
import numpy as np
import pandas as pd
from kg_base import KG_base
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split as sk_tts
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.metrics import r2_score as sk_r2
from sklearn.model_selection import cross_val_score as sk_cross

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
    
    def __init__(self):
        super().__init__()
        self.m = None
        self.m_path = "regression_model.pkl"
    
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

        # Pré-traitement #
        #----------------#
    def preprocess(self, x, y, columns=['distr_noise'],
                         select=True, verbose=True):
        """Variables catégorielles et équilibrage.
        Note : on assume que le jeu de données est propre."""
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
            # print(ix)
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
            # Passage au DataFrame?
        x = pd.DataFrame(x, columns=names[1:])
        y = pd.DataFrame(y, columns=[names[0]])
        return x, y
    
        # Fit/predict #
        #-------------#
    def fit(self, x, y, verbose=False):
        """Entraîne le modèle.
        Note : on ne peut pas encore 'continuer' l'entraînement du modèle."""
        self.m = LinearRegression()           # initialisation
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
    def eval(self, x, y, verbose=True):
        """Évalue le modèle déjà entraîné, sans validation croisée."""
        yp = self.predict(x)                  # évaluation
        return sk_mse(y, yp), sk_r2(y, yp)    # MSE et R2

        # Méthode principale #
        #--------------------#
    def test(self, x, y, preprocess=True, verbose=True):
        """Méthode générale pour tester le modèle 
        à partir d'un jeu de données."""
        if preprocess:                            # pré-traitement
            x, y = self.preprocess(x, y, verbose=False)
        x_tr, x_val, y_tr, y_val = sk_tts(x, y)   # entraînement/validation
        self.m = LinearRegression()
        sc = sk_cross(self.m, x_tr, y_tr, cv=5)   # validation croisée
        self.fit(x_tr, y_tr, verbose=False)       # entraînement général
        mse, r2 = self.eval(x_val, y_val)         # évaluation générale
        if verbose:
            self.log(
                "Model Performance:\n"+
                f"Cross-validation mean: {sc.mean():.02f} "
                f", std: {sc.std():.02f}\n"+
                f"Mean Square Error: {mse:.02f}\n"+
                f"R² Score: {r2:.02f}"
            )
        return sc, mse, r2

if __name__ == "__main__":
    from kg_test import KG_test
    dat = KG_test().load("kg_test.xlsx")
    kg = KG_model()
    kg.test(dat[:,1:], dat[:,0], verbose=True)
    