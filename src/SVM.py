import torch as th

class SVM:
    def __init__(self, C=1.0, lr=0.001, n_iters=1000):
        self.C = C  
        self.lr = lr  
        self.n_iters = n_iters  

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize Lagrange multipliers
        self.alpha = th.zeros(n_samples)

        # Kernel matrix (linear kernel)
        K = X@X.T

        # Gradient ascent to maximize the dual objective
        for _ in range(self.n_iters):
            for i in range(n_samples):
                gradient = 1 - y[i] * th.sum(self.alpha * y * K[:, i])
                self.alpha[i] += self.lr * gradient

        sv = self.alpha > 1e-5
        self.sv_alpha = self.alpha[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]

        self.w = th.sum(self.sv_alpha.unsqueeze(1) * self.sv_y.unsqueeze(1) * self.sv_X, axis=0)
        self.b = th.mean(self.sv_y - self.sv_X@self.w)

    def predict(self, X):
        return th.sign(self.w@X.T + self.b)

    def decision_function(self, X):
        return self.w@X.T + self.b

class OneVsAllSVM:
    def __init__(self, num_classes, C=1.0, kernel='linear'):
        """
        Initializes the One-vs-All SVM classifier.
        
        Parameters:
        - SVM_class: The SVM classifier class (assumed to be already implemented).
        - num_classes: The number of classes in the classification problem.
        - C: The regularization parameter for the SVM.
        - kernel: The kernel function for the SVM.
        """
        self.num_classes = num_classes
        self.C = C
        self.kernel = kernel
        self.models = []

    def fit(self, X, y):
        """
        Trains a One-vs-All SVM classifier.
        
        Parameters:
        - X: Feature matrix (n_samples x n_features)
        - y: Target vector (n_samples,)
        """
        for class_label in range(0, self.num_classes):
            # Convert the labels for this class vs all others
            y_binary = th.where(y == class_label, 1, -1)
            
            # Train the SVM classifier for the current class
            svm = SVM(C=self.C)#, kernel=self.kernel)
            svm.fit(X, y_binary)
            self.models.append(svm)

    def predict(self, X):
        """
        Predicts the class labels for ithut samples using the trained One-vs-All SVMs.
        
        Parameters:
        - X: Feature matrix (n_samples x n_features)
        
        Returns:
        - Predicted class labels for each sample.
        """
        # Get decision values from each model
        decision_values = th.tensor([model.decision_function(X) for model in self.models])
        
        # Get the predicted class by selecting the class with the highest decision value
        predictions = th.argmax(decision_values, axis=0)  # Adding 1 to match class labels (1-based)
        return predictions
