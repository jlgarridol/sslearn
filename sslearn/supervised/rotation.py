import numpy as np
from sklearn.decomposition import PCA
import sklearn.utils
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier


class Rotation(TransformerMixin, BaseEstimator):
    
    def __init__(self, group_size=3, group_weight=.5, pca=PCA(), random_state=None):
        """Rotation Transformer.

        Join of subspaces of the dataset transfomed with PCA keeping the subspace dimensionality.

        The transformation is made with a subsamples of the subspace. The subsamples and the subspaces are made randomly.

        Args:
            group_size (int, optional): Number of the features for each subspace. If the number of features mod group size are not zero, then the last subspace are created with features used previously selected randomly.. Defaults to 3.
            group_weight (float, optional): Percent of the instances to use to fit the PCA for each subspace. Defaults to .5.
            pca (PCA, optional): PCA configuration, the n_components will be overwritten. Defaults to PCA().
            random_state (None int or RandomState, optional): Random state for create subspaces and subsamples. Defaults to None.
        """
        self.group_size=group_size
        self.random_state=sklearn.utils.check_random_state(random_state)
        self.group_weight=group_weight
        self.pca = pca
        self.groups = []
        self.pcas = []
        
    def fit(self, X, y=None):
        """Create a rotation.

        Args:
            X (array-like, shape (n_samples, n_features)): Training data, where n_samples is the number of samples and n_features is the number of features.
            y (None): Ignored variable.
        """
        rows, cols = X.shape
        cl = list(range(cols))
        rl = list(range(rows))
        # Primero se randomizan las columnas
        self.random_state.shuffle(cl)
        # Se generan las columnas para los grupos
        idx = 0
        while idx < len(cl):
            gr = []
            for i in range(self.group_size):
                if i+idx >= len(cl):
                    gr.append(self.random_state.choice(cl))
                else:
                    gr.append(cl[i+idx])
            
            self.groups.append(gr)
            idx += self.group_size
        # Se han generado los grupos de las columnas
        # Con esto se crean subconjuntos de los datos
        groups_X = []
        for g in self.groups:
            groups_X.append(X[:,g])
        # De cada grupo con sus datos se escogen aleatoriamente el porcentaje deseado
        groups_T = []
        for g in groups_X:
            # Se "barajean" las filas
            self.random_state.shuffle(rl)
            # Se escogen las primeras (group_weight*total_columnas)
            sel = int(self.group_weight*rows)
            groups_T.append(g[0:sel,:])
            
        # Una vez se tienen los objetos para entrenar entonces se crean los PCA
        for g in groups_T:
            p = clone(self.pca)
            p.n_componentes_ = self.group_size
            #p = PCA(self.group_size)
            p.fit(g)
            self.pcas.append(p)
            
        # Ya está el modelo entrenado            
        
            
    def transform(self, X):
        """Apply rotation to X.

        X rotated in each subspace and then the rotated subspaces are joined to create the global rotation of X.

        Args:
            X (array-like, shape (n_samples, n_features)): New data, where n_samples is the number of samples and n_features is the number of features.

        Returns:
            array-like, shape (n_samples, n_components): Transformed values.
        """
        assert len(self.pcas) > 0 and len(self.pcas) == len(self.groups), "No se ha generado al transformación"
        # Se crean los subconjuntos de los datos transformados
        tformed = []
        for i in range(len(self.pcas)):
            pca = self.pcas[i]
            group = self.groups[i]
            x_n = X[:,group]
            x_t = pca.transform(x_n)
            tformed.append(x_t)
            
        return np.concatenate(tformed,axis=1)
            
    def fit_transform(self, X, y=None):
        """Create the rotation of X and get the rotation of X.

        Args:
            X (array-like, shape (n_samples, n_features)): Training data, where n_samples is the number of samples and n_features is the number of features.
            y (None): Ignored variable.

        Returns:
            array-like, shape (n_samples, n_components): Rotation of X.
        """
        self.fit(X)
        return self.transform(X)
            

class RotatedTree(ClassifierMixin, BaseEstimator):


    def __init__(self, base_estimator=DecisionTreeClassifier(), rotation=Rotation()):
        """Create a rotation and train a decision tree classifier.

        Args:
            base_estimator (object, optional): The base estimator to fit on rotation of the dataset. If None, then the base estimator is a decision tree. Defaults to DecisionTreeClassifier().
            rotation (Rotation, optional): The configured rotation transform. Defaults to Rotation().
        """
        self.base_estimator = clone(base_estimator)
        self.rotation = clone(rotation)

    def fit(self, X, y):
        """Fit the Rotated Tree model.

        Args:
            X (array-like, shape (n_samples, n_features)): Training data, where n_samples is the number of samples and n_features is the number of features.
            y (array-like, shape (n_samples,)): The target values.
        """
        X = self.rotation.fit_transform(X)
        self.base_estimator.fit(X,y)
        self.classes_=self.base_estimator.classes_

    def predict(self, X):
        """Predict class for X.

        Args:
            X (array-like, shape (n_samples, n_features)): The input samples.

        Returns:
            ndarray of shape (n_samples,): The predicted classes.
        """
        X = self.rotation.transform(X)
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for X.


        Args:
            X (array-like, shape (n_samples, n_features)): The input samples.

        Returns:
            array-like, shape (n_samples, n_features): The class probabilities of the input samples. The order of the classes corresponds to that in the attribute classes_.
        """
        X = self.rotation.transform(X)
        return self.base_estimator.predict_proba(X)

class RotationForestClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=10, min_group_size=3, max_group_size=3, rotation=Rotation(), random_state=None):
        """Create a ensemble of rotation trees for clasification.

        Args:
            base_estimator (object, optional): The base estimator to fit on each rotation of the dataset. If None, then the base estimator is a decision tree. Defaults to DecisionTreeClassifier().
            n_estimators (int, optional): Number of estimators in the ensemble. Defaults to 10.
            min_group_size (int, optional): Min group of the features for subspaces. Defaults to 3.
            max_group_size (int, optional): Max group of the features for subspaces. Defaults to 3.
            rotation (Rotation, optional): Configuration for a rotation. The random_state and group_size will be overwritten in each iteration. Defaults to Rotation().
            random_state (None int or RandomState, optional): Random state for create subspaces and subsamples. Defaults to None.
        """
        self.random_state=sklearn.utils.check_random_state(random_state)
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.base_estimator = base_estimator
        self.rotation = rotation
        self.rotation.rand = self.random_state
        self.n_estimators = n_estimators

    def fit(self, X, y):
        """Fit the RotationForest model.

        Args:
            X (array-like, shape (n_samples, n_features)): Training data, where n_samples is the number of samples and n_features is the number of features.
            y (array-like, shape (n_samples,)): The target values.
        """
        self.estimators = []
        for _ in range(self.n_estimators):
            size = self.random_state.randint(self.min_group_size, self.max_group_size+1)
            self.rotation.group_size = size
            self.rotation.random_state = self.random_state
            tree = RotatedTree(self.base_estimator, self.rotation)
            tree.fit(X, y)
            self.estimators.append(tree)
        self.classes_ = self.estimators[0].classes_
    
    def predict(self, X):
        """Predict class for X.

        Args:
            X (array-like, shape (n_samples, n_features)): The input samples.

        Returns:
            ndarray of shape (n_samples,): The predicted classes.
        """
        classes = []
        for t in self.estimators:
            predicts = t.predict(X)
            classes.append(predicts)
        # Vote
        classes = np.array(classes)
        (v, c) = np.unique(classes, return_counts=True, axis=0)
        ind=np.argmax(c, axis=0)
        return v[ind]

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The probability for each instance is the mean between all classifiers in ensemble.


        Args:
            X (array-like, shape (n_samples, n_features)): The input samples.

        Returns:
            array-like, shape (n_samples, n_features): The class probabilities of the input samples. The order of the classes corresponds to that in the attribute classes_.
        """
        probas = []
        for t in self.estimators:
            predicts = t.predict_proba(X)
            probas.append(predicts)
        probas = np.array(probas)
        return probas.mean(axis=0)
