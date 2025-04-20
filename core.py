# Import Libraries
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import time


""" [1] Defining the Optimization Main Class """
""" ---------------------------------------- """
class HyperparameterOptimizer:
    
    def __init__(self, obj_func, params, scoring, opt_type="max", cv=5, verbose=1):
        """
        The function defines a HyperparameterOptimizer instance.
        
        Inputs:
            obj_func: The machine learning model or pipeline being created.
            opt_type (str): Type of optimization: "max" for maximization (default) or "min" for minimization.
            params (dict): A dictionary of parameters for which the model is to be optimized.
            scoring (str): The method based on which the machine learning model is evaluated.
            cv (int): The number of cross-validation splits.
            verbose (int): A binary with a value of 1 (default) to show iteration information.
        """
        
        self.obj_func = obj_func
        self.params = params
        self.scoring = scoring
        self.opt_type = opt_type
        self.cv = cv
        self.verbose = verbose
    
    
    def optimizePS(self, features, target, nParticles, bounds, w=0.5, c1=1, c2=1, maxIter=20, mutation_prob=0.1):
        """
        The function utilizes Particle Swarm (PS) optimization for finding the optimal values of the machine learning (ML) algorithm/pipeline hyperparameters. 
        
        Args:
            features: A dataset containing the input features for the ML algorithm/pipeline.
            target: The target variable of the ML algorithm/pipeline.
            nParticles (int): The number of particles in the swarm.
            bounds (1darray): The bounds of the hyperparameter values [(x1_min, x1_max), (x2_min, x2_max), ... ]. The array dimension should be equal to the number of hyperparameters.
            w (float): The inertia weight.
            c1 (float): The cognitive weight.
            c2 (float): The social weight.
            maxIter (int): The maximum number of iterations.
            mutation_prob (float): The probability of mutation for non-numeric hyperparameters.
            
        Outputs:
            population (Dataframe): The population used in the algorithm with their locations and values.
            Gbest_historry (ndarray): Values of optimal solutions in all the iterations.
            Gbest_pos (ndarray): The position of the optimal solution.
            Gbest_score (float): The score of the optimal solution.
        """

        # 1. Start Optimization Time Recording
        print("PS hyperparameter optimization has started . . .")
        start_time = time.time()
        
        
        # 2. Initialize Gbest Position and Score
        print("Initializing Global Best Position and Score . . .")
        Gbest_pos = None                                                        # Temporary Position of the Global Best Solution
        if self.opt_type=="max":                                                # Maximization Problem
            Gbest_score = -1 * np.inf
        else:                                                                   # Minimization Problem
            Gbest_score = np.inf
        
        # 3. Initialize Particles
        print("Initializing Swarm Particles . . .")
        bounds = np.array(bounds, dtype=object)
        particles = []
        for _ in range(nParticles):
            position = np.array([
                                    np.random.choice(b) if isinstance(b, (list, np.ndarray)) else  # If it's a list or array, choose randomly from it
                                    np.random.uniform(low=b[0], high=b[1]) if isinstance(b[0], float) or isinstance(b[1], float) else  # If it's a tuple of floats, treat as float range
                                    np.random.randint(low=b[0], high=b[1] + 1)  # If it's a tuple of integers, treat as integer range
                                    for b in bounds
                                ], dtype=object)
            
            velocity = np.array([
                                    np.random.choice(b) if isinstance(b, (list, np.ndarray)) else  # If it's a list or array, choose randomly from it
                                    np.random.uniform(low=-(b[1] - b[0]), high=(b[1] - b[0])) if isinstance(b[0], float) or isinstance(b[1], float) else  # If it's a tuple of floats, treat as float range
                                    np.random.randint(low=-(b[1] - b[0]), high=(b[1] - b[0]) + 1)  # If it's a tuple of integers, treat as integer range
                                    for b in bounds
                                ], dtype=object)
            
            score = self.eval_objfunc(position, features, target)
            particle = {
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_score': score
            }
            particles.append(particle)
            # Update Gbest Position and Score
            if (score > Gbest_score) and (self.opt_type=="max"):                 # Maximization Problem
                Gbest_pos = position
                Gbest_score = score
            elif (score < Gbest_score) and (self.opt_type=="min"):               # Minimization Problem
                Gbest_pos = position
                Gbest_score = score
            
        # 4. Record Iterations' Optimal Gbest Values and Set Variable Values
        Gbest_history = []                                                       # Keep Track of Objective Function Value of Each Iteration
        
        # 4.1. Initialize Iterator
        iterator = 0
        
        # 5. Optimization Main Loop
        print("Starting Optimization Main Loop . . .")
        while iterator < maxIter:
            
            # 5.1. Setting the Actual Number of Iterations Variable
            iterator += 1
            
            # 5.2. Update Particle Values
            for particle in particles:
                
                # 5.2.1. Initialize New Velocity and Position
                new_velocity = np.zeros(len(particle['velocity']), dtype=object)
                new_position = np.zeros(len(particle['position']), dtype=object)
                
                # 5.2.2. Update Velocity and Position
                for i in range(len(particle['position'])):
        
                    if isinstance(bounds[i], (list, np.ndarray)):  # Non-numeric types (strings, booleans, lists)
                        # Skip velocity update for non-numeric types
                        new_velocity[i] = 0  # Reset velocity to 0

                        # Update position by randomly selecting a new value from the list
                        if np.random.rand() < mutation_prob:  # Apply mutation with probability
                            new_position[i] = np.random.choice(bounds[i])
                        else:
                            new_position[i] = particle['position'][i]  # Keep the current value
                    
                    else:
                        # Calculate cognitive and social components
                        r1, r2 = np.random.rand(2)
                        cognitive = c1 * r1 * (particle['best_position'][i] - particle['position'][i])
                        social = c2 * r2 * (Gbest_pos[i] - particle['position'][i])
                        
                        # Update velocity
                        new_velocity[i] = w * particle['velocity'][i] + cognitive + social
                        
                        # Clip velocity to bounds
                        if isinstance(bounds[i][0], float) or isinstance(bounds[i][1], float):  # Float range
                            new_velocity[i] = np.clip(new_velocity[i], -np.abs(bounds[i][1] - bounds[i][0]), np.abs(bounds[i][1] - bounds[i][0]))
                        else:  # Integer range
                            new_velocity[i] = np.clip(new_velocity[i], -np.abs(bounds[i][1] - bounds[i][0]), np.abs(bounds[i][1] - bounds[i][0]))
                            new_velocity[i] = int(new_velocity[i])
                        
                        # Update position
                        new_position[i] = particle['position'][i] + new_velocity[i]
                        
                        # Clip position to bounds
                        if isinstance(bounds[i][0], float) or isinstance(bounds[i][1], float):  # Float range
                            new_position[i] = np.clip(new_position[i], bounds[i][0], bounds[i][1])
                        else:  # Integer range
                            new_position[i] = np.clip(new_position[i], bounds[i][0], bounds[i][1])
                            new_position[i] = int(new_position[i])
                
                # Fully Update Particle's Velocity and Position
                particle['velocity'] = new_velocity
                particle['position'] = new_position


                # 5.2.3. Evaluate New Position
                score = self.eval_objfunc(particle['position'], features, target)
                
                # 5.2.4. Update Pbest
                if (score > particle['best_score']) and (self.opt_type=="max"):     # Maximization Problem
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()
                elif (score < particle['best_score']) and (self.opt_type=="min"):   # Minimization Problem
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()
                
                # 5.2.5. Update Gbest
                if (score > Gbest_score) and (self.opt_type=="max"):                # Maximization Problem
                    Gbest_pos = particle['position'].copy()
                    Gbest_score = score
                elif (score < Gbest_score) and (self.opt_type=="min"):   # Minimization Problem
                    Gbest_pos = particle['position'].copy()
                    Gbest_score = score
            
            # 5.3. Record Gbest Score
            Gbest_history.append(Gbest_score)
            
            # 5.4. Printing
            if self.verbose == 1:
                print("Iteration #{}".format(iterator))
                print("Corresponding Optimal Solution: {}".format(Gbest_pos))
                print("Corresponding Optimum: {}".format(Gbest_score))
                print("------")
        
        # 6. Storing Optimization Variables in a Dictionary
        self.PS_attritbutes = {
                                "Number of Iterations": iterator,
                                "Number of Particles": nParticles,
                                "Inertia Weight": w,
                                "Cognitive Weight": c1,
                                "Social Weight": c2
                                }
        
        # 7. Storing Optimization Results in Callable Variables
        self.PS_Particles = particles
        self.PS_iterOptSols = {
                                "Iterations' Index": np.arange(1, iterator+1),
                                "Iteration's Optimum Value": Gbest_history
                               }
        self.PS_finalOptimalSol = {
                                    "Optimal Solution Variables' Values": Gbest_pos,
                                    "Optimal Solution Value": Gbest_score
                                    }
        
        # Print a Closure Message
        end_time = time.time()
        total_exec_time = end_time - start_time
        print("Optimization Finished Successfully!")
        print("Optimization Concluded in {} Seconds.".format(np.round(total_exec_time,3)))
        
        
        return particles, Gbest_history, Gbest_pos, Gbest_score
    

    # Objective Function Evaluation Function
    def eval_objfunc(self, dec_vars, features, target):
        """
        The function is used to evaluate the objective function value in a Genetic Algorithm optimization process.

        Inputs:
            dec_vars (ndarray): A list of decision variables that represent the parameters to be optimized for the objective function.
            features: A matrix of features that are used as inputs for the objective function.
            target: A vector of target values that the objective function aims to predict.
            
        Outputs:
            cv_accuracy_score (float): Accuracy of the objective function.
        """
        
        # Defining Cross-validation Object
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=1)
        
        # Augment the Pipeline with Parameters from Decision Variables
        p_index = 0
        for key, value in self.params.items():
            if key in self.obj_func.get_params():
                self.obj_func.set_params(**{key: dec_vars[p_index]})
                p_index += 1
        
        # Evaluate Objective Function Value through Cross-validation [Time Consuming!]
        cv_accuracy_score = cross_val_score(
                                            self.obj_func,
                                            features,
                                            target,
                                            cv=kf,
                                            scoring= self.scoring,
                                            error_score="raise",
                                            n_jobs=-1
                                            ).mean()
                
        return cv_accuracy_score