import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import joblib
import random
from utils.data_loader import DataLoader

# Create fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

class GeneticOptimizer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.toolbox = base.Toolbox()
        self.setup_toolbox()
        
    def setup_toolbox(self):
        """Setup DEAP toolbox"""
        # Individual: binary array for feature selection + hyperparameters
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("attr_int", random.randint, 10, 200)  # n_estimators
        self.toolbox.register("attr_depth", random.randint, 5, 30)  # max_depth
        
        def create_individual():
            # Feature selection (binary) + hyperparameters
            features = [self.toolbox.attr_bool() for _ in range(self.n_features)]
            n_estimators = self.toolbox.attr_int()
            max_depth = self.toolbox.attr_depth()
            return features + [n_estimators, max_depth]
        
        self.toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_individual, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def evaluate_individual(self, individual):
        """Evaluate fitness of an individual"""
        # Extract features and hyperparameters
        feature_mask = individual[:self.n_features]
        n_estimators = individual[self.n_features]
        max_depth = individual[self.n_features + 1]
        
        # Check if at least one feature is selected
        if sum(feature_mask) == 0:
            return (0.0,)
        
        # Select features
        X_selected = self.X.iloc[:, [i for i, selected in enumerate(feature_mask) if selected]]
        
        # Create and evaluate model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        # Use cross-validation for robust evaluation
        scores = cross_val_score(model, X_selected, self.y, cv=3, scoring='accuracy')
        return (np.mean(scores),)
    
    def mutate_individual(self, individual, indpb):
        """Custom mutation function"""
        # Mutate feature selection
        for i in range(self.n_features):
            if random.random() < indpb:
                individual[i] = 1 - individual[i]
        
        # Mutate hyperparameters
        if random.random() < indpb:
            individual[self.n_features] = random.randint(10, 200)
        if random.random() < indpb:
            individual[self.n_features + 1] = random.randint(5, 30)
        
        return (individual,)
    
    def optimize(self, population_size=50, generations=20):
        """Run genetic algorithm optimization"""
        print("Starting genetic algorithm optimization...")
        
        # Create initial population
        population = self.toolbox.population(n=population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run algorithm
        population, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=0.7, mutpb=0.3,
            ngen=generations,
            stats=stats,
            verbose=True
        )
        
        # Get best individual
        best_individual = tools.selBest(population, 1)[0]
        best_features = [i for i, selected in enumerate(best_individual[:self.n_features]) if selected]
        best_params = {
            'n_estimators': best_individual[self.n_features],
            'max_depth': best_individual[self.n_features + 1]
        }
        
        print(f"Best fitness: {best_individual.fitness.values[0]:.4f}")
        print(f"Selected features: {len(best_features)}/{self.n_features}")
        print(f"Best parameters: {best_params}")
        
        return best_features, best_params, logbook

def run_genetic_optimization():
    """Run genetic algorithm for feature selection and hyperparameter tuning"""
    print("Loading data for genetic optimization...")
    loader = DataLoader()
    X, y, y_binary = loader.load_nsl_kdd()
    
    # Use a subset for faster optimization
    sample_size = min(5000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[indices]
    y_sample = y_binary.iloc[indices]
    
    # Run optimization
    optimizer = GeneticOptimizer(X_sample, y_sample)
    best_features, best_params, logbook = optimizer.optimize()
    
    # Save results
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'best_features': best_features,
        'best_params': best_params,
        'optimization_history': logbook
    }, 'models/genetic_optimization_results.pkl')
    
    print("Genetic optimization completed and saved!")
    return best_features, best_params

if __name__ == "__main__":
    run_genetic_optimization()
