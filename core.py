# Import Libraries
from sklearn.model_selection import KFold, cross_val_score
import itertools
import numpy as np
import time


""" [1] Define the Core Hyperparameter Optimization Class """
class HyperparameterOptimizer:
    
    # [1] Initialization Method
    def __init__(self, obj_func, params, scoring, opt_type="max", cv=5, verbose=1):
        """
        Initialize the HyperparameterOptimizer class with configuration settings.

        Args:
            obj_func: The machine learning model or pipeline to be optimized.
            params (dict): Dictionary of hyperparameters and their search ranges.
            scoring (str): Performance metric for evaluation.
            opt_type (str): Optimization mode - "max" for maximization or "min" for minimization.
            cv (int): Number of cross-validation folds.
            verbose (int): Flag for verbosity (1 to enable logging).
        """
        
        self.obj_func = obj_func
        self.params = params
        self.scoring = scoring
        self.opt_type = opt_type
        self.cv = cv
        self.verbose = verbose
    
    
    # [2] Particle Swarm Optimization Method
    def optimizePSO(self, features, target, nParticles, bounds, w=0.5, c1=1, c2=1, maxIter=20, mutation_prob=0.1):
        """
        The function utilizes Particle Swarm (PSO) optimization for finding the optimal values of the machine learning (ML) algorithm/pipeline hyperparameters. 
        
        Args:
            features: A dataset containing the input features for the ML algorithm/pipeline.
            target: The target variable of the ML algorithm/pipeline.
            nParticles (int): The number of particles in the swarm.
            bounds (1darray): The bounds of the hyperparameter values [(x1_min, x1_max), (x2_min, x2_max), ... ]. The array dimension should be equal to the number of hyperparameters.
            w (float): The inertia weight.
            c1 (float): The cognitive component weight.
            c2 (float): The social component weight.
            maxIter (int): The maximum number of optimization iterations.
            mutation_prob (float): The probability of mutation for categorical parameters.
            
        Outputs:
            population (Dataframe): The population used in the algorithm including their positions and values.
            best_history (ndarray): Values of optimal solutions in all the iterations.
            best_pos (ndarray): The position of the optimal solution.
            best_score (float): The score of the optimal solution.
        """

        # 1. Start Optimization Time Recording
        print("PS hyperparameter optimization has started . . .")
        start_time = time.time()
        
        
        # 2. Initialize Gbest Position and Score
        print("Initializing Global Best Position and Score . . .")
        Gbest_pos = None                                                                # Temporary Position of the Global Best Solution
        if self.opt_type=="max":                                                        # Maximization Problem
            Gbest_score = -1 * np.inf
        else:                                                                           # Minimization Problem
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
            if (score > Gbest_score) and (self.opt_type=="max"):                        # Maximization Problem
                Gbest_pos = position
                Gbest_score = score
            elif (score < Gbest_score) and (self.opt_type=="min"):                      # Minimization Problem
                Gbest_pos = position
                Gbest_score = score
        

        # 4. Record Iterations' Optimal Gbest Values and Set Variable Values
        Gbest_history = []                                                              # Keep Track of Objective Function Value of Each Iteration
        
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
        
                    if isinstance(bounds[i], (list, np.ndarray)):                       # Non-numeric Types (strings, booleans, lists)
                        # Skip Velocity Update for Non-numeric Types
                        new_velocity[i] = 0                                             # Reset Velocity to 0

                        # Update Position by Randomly Selecting a New Value from the List
                        if np.random.rand() < mutation_prob:                            # Apply Mutation with Probability
                            new_position[i] = np.random.choice(bounds[i])
                        else:
                            new_position[i] = particle['position'][i]                   # Keep the Current Value
                    
                    else:
                        # Calculate Cognitive and Social Components
                        r1, r2 = np.random.rand(2)
                        cognitive = c1 * r1 * (particle['best_position'][i] - particle['position'][i])
                        social = c2 * r2 * (Gbest_pos[i] - particle['position'][i])
                        
                        # Update Velocity
                        new_velocity[i] = w * particle['velocity'][i] + cognitive + social
                        
                        # Clip velocity to Bounds
                        if isinstance(bounds[i][0], float) or isinstance(bounds[i][1], float):  # Float range
                            new_velocity[i] = np.clip(new_velocity[i], -np.abs(bounds[i][1] - bounds[i][0]), np.abs(bounds[i][1] - bounds[i][0]))
                        else:                                                           # Integer Range
                            new_velocity[i] = np.clip(new_velocity[i], -np.abs(bounds[i][1] - bounds[i][0]), np.abs(bounds[i][1] - bounds[i][0]))
                            new_velocity[i] = int(new_velocity[i])
                        
                        # Update Position
                        new_position[i] = particle['position'][i] + new_velocity[i]
                        
                        # Clip Position to Bounds
                        if isinstance(bounds[i][0], float) or isinstance(bounds[i][1], float):  # Float Range
                            new_position[i] = np.clip(new_position[i], bounds[i][0], bounds[i][1])
                        else:                                                                   # Integer Range
                            new_position[i] = np.clip(new_position[i], bounds[i][0], bounds[i][1])
                            new_position[i] = int(new_position[i])
                
                # Fully Update Particle's Velocity and Position
                particle['velocity'] = new_velocity
                particle['position'] = new_position


                # 5.2.3. Evaluate New Position
                score = self.eval_objfunc(particle['position'], features, target)
                
                # 5.2.4. Update Pbest
                if (score > particle['best_score']) and (self.opt_type=="max"):         # Maximization Problem
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()
                elif (score < particle['best_score']) and (self.opt_type=="min"):       # Minimization Problem
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()
                
                # 5.2.5. Update Gbest
                if (score > Gbest_score) and (self.opt_type=="max"):                    # Maximization Problem
                    Gbest_pos = particle['position'].copy()
                    Gbest_score = score
                elif (score < Gbest_score) and (self.opt_type=="min"):                  # Minimization Problem
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
        self.PSO_attritbutes = {
                                    "Number of Iterations": iterator,
                                    "Number of Particles": nParticles,
                                    "Inertia Weight": w,
                                    "Cognitive Weight": c1,
                                    "Social Weight": c2
                                }
        

        # 7. Storing Optimization Results in Callable Variables
        self.PSO_Particles = particles
        self.PSO_iterOptSols = {
                                    "Iterations' Index": np.arange(1, iterator+1),
                                    "Iteration's Optimum Value": Gbest_history
                               }
        self.PSO_finalOptimalSol = {
                                        "Optimal Solution Variables' Values": Gbest_pos,
                                        "Optimal Solution Value": Gbest_score
                                    }
        
        # Print a Closure Message
        end_time = time.time()
        total_exec_time = end_time - start_time
        print("Optimization Finished Successfully!")
        print("Optimization Concluded in {} Seconds.".format(np.round(total_exec_time,3)))
        
        
        return particles, Gbest_history, Gbest_pos, Gbest_score


    # [3] Pattern Search Optimization
    def optimizePS(self, features, target, bounds, mesh_size_coeff, acc_coeff, contr_coeff, search_method='gps', min_mesh_ratio=0.001, maxIter=20):
        """
        The function utilizes Particle Swarm (PS) optimization for finding the optimal values of the machine learning (ML) algorithm/pipeline hyperparameters. 
        
        Args:
            features: A dataset containing the input features for the ML algorithm/pipeline.
            target: The target variable of the ML algorithm/pipeline.
            bounds (1darray): The bounds of the hyperparameter values [(x1_min, x1_max), (x2_min, x2_max), ... ]. The array dimension should be equal to the number of hyperparameters.
            mesh_size_coeff (float): The mesh size coefficient. For each dimension x_n, (1) when mesh_size_coeff >= 1: mesh_size_n = mesh_size_coeff * x_n_min
                                                                                        (2) when mesh_size_coeff < 1: mesh_size_n = mesh_size_coeff * x_n_max
            acc_coeff (float): The acceleration/mesh expansion coefficient.
            contr_coeff (float): The mesh contraction coefficient. Between [0, 1] and determines the percentage by which the mesh_size will be reduced if no better solutions are found.
            search_method (str): 'gps' for Generalized Pattern Search (in all directions).
                                 'mads' for Mesh Adaptive Direct Search (in positive and all-negative directions).
            min_mesh_ratio (float): Minimum mesh size as a percentage of parameter range.
            maxIter (int): The maximum number of iterations.
            
        Outputs:
            best_history (ndarray): Values of optimal solutions in all the iterations.
            best_pos (ndarray): The position of the optimal solution.
            best_score (float): The score of the optimal solution.
        """

        # 1. Start Optimization Time Recording
        print("Pattern Search hyperparameter optimization has started . . .")
        start_time = time.time()
        
        
        # 2. Initialize Best Position and Score
        best_pos = None                                                                 # Temporary Position of the Global Best Solution
        if self.opt_type=="max":                                                        # Maximization Problem
            best_score = -1 * np.inf
        else:                                                                           # Minimization Problem
            best_score = np.inf
        

        # 3. Initialize the Algorithm
        print("Initializing Pattern Search . . .")

        # 3.2. Define Initial Mesh Size
        mesh_size = np.array([
                                    1 if isinstance(b, (list, np.ndarray)) else                                                                                             # If it's a list or array, assign a value of 1 to mesh size
                                    (mesh_size_coeff*b[0] if mesh_size_coeff>=1 else mesh_size_coeff*b[1]) if isinstance(b[0], float) or isinstance(b[1], float) else       # If it's a tuple of floats, treat as float range
                                    int(mesh_size_coeff*b[0] if mesh_size_coeff>=1 else mesh_size_coeff*b[1])                                                                                              # If it's a tuple of integers, treat as integer range
                                    for b in bounds
                            ], dtype=object)

        # 3.3. Initialize Base Point
        base_point_pos = np.array([
                                    np.random.choice(b) if isinstance(b, (list, np.ndarray)) else                                               # If it's a list or array, choose randomly from it
                                    np.random.uniform(b[0], b[1]) if isinstance(b[0], float) or isinstance(b[1], float) else           # If it's a tuple of floats, treat as float range
                                    np.random.randint(b[0], b[1] + 1)                                                                  # If it's a tuple of integers, treat as integer range
                                    for b in bounds
                            ], dtype=object)
        base_point_score = self.eval_objfunc(base_point_pos, features, target)

        # 3.4. Update Best Position and Score
        best_pos, best_score = base_point_pos.copy(), base_point_score
        

        # 4. Record Iterations' Optimal Gbest Values and Set Variable Values
        best_history = []                                                                                                               # Keep Track of Objective Function Value of Each Iteration
        
        # 4.1. Initialize Iterator
        iterator = 0
        

        # 5. Optimization Main Loop
        print("Starting Optimization Main Loop . . .")
        while iterator < maxIter:
            
            # 5.1. Setting the Actual Number of Iterations Variable
            iterator += 1
            improved = False
            
            # 5.2. Generate Pattern Vectors
            pattern_vectors = []
            for i, b in enumerate(bounds):
                if isinstance(b, (list, np.ndarray)):                                   # Categorical Parameter
                    current_idx = list(b).index(base_point_pos[i])
                    pattern_vectors.append([
                                                (i, 1),                                 # Next option
                                                (i, -1)                                 # Previous option
                                            ])
                else:                                                                   # Numerical Parameter
                    if search_method == 'mads':
                        # MADS: Positive and Fully Negative Directions
                        pattern_vectors.append([0, 1])
                    else:
                        # GPS: Positive and Negative Vectors
                        pattern_vectors.append([-1, 0, 1])

            # 5.3. Generate All Directions
            all_directions = []
            if search_method == 'mads':
                positive_directions = list(itertools.product(*[[0, 1] if not isinstance(b, (list, np.ndarray)) else [0, 1] for b in bounds]))
                positive_directions = [d for d in positive_directions if sum(d) == 1]  # Only unit vectors
                negative_direction = tuple([-1 if not isinstance(b, (list, np.ndarray)) else 0 for b in bounds])
                all_directions = positive_directions + [negative_direction]
            else:
                all_directions = list(itertools.product(*[[-1, 0, 1] if not isinstance(b, (list, np.ndarray)) else [-1, 0, 1] for b in bounds]))
                all_directions = [d for d in all_directions if sum(abs(x) for x in d) == 1]  # Only unit vectors

            # 5.4. Add Categorical Directions (replace numerical 0s with categorical moves)
            final_directions = []
            for direction in all_directions:
                final_dir = []
                cat_moves = []
                has_numerical_move = False
                
                for dim, move in enumerate(direction):
                    if isinstance(bounds[dim], (list, np.ndarray)):                     # Categorical
                        if move != 0:
                            cat_moves.append((dim, 1 if move > 0 else -1))
                        final_dir.append(0)                                             # Placeholder
                    else:                                                               # Numerical
                        final_dir.append(move)
                        if move != 0:
                            has_numerical_move = True
                
                if cat_moves and not has_numerical_move:
                    # Pure Categorical Move
                    for cat_dim, delta in cat_moves:
                        new_dir = final_dir.copy()
                        new_dir[cat_dim] = delta
                        final_directions.append(tuple(new_dir))
                else:
                    # Numerical or Mixed Move
                    final_directions.append(tuple(final_dir))

            final_directions = list(set([d for d in final_directions if any(x != 0 for x in d)]))

            # 5.5. Evaluate All Directions
            for direction in final_directions:
                trial = base_point_pos.copy()
                for dim, move in enumerate(direction):
                    if isinstance(bounds[dim], (list, np.ndarray)):  # Categorical
                        if move != 0:
                            options = bounds[dim]
                            current_idx = list(options).index(trial[dim])
                            trial[dim] = options[(current_idx + move) % len(options)]
                    else:                                                               # Numerical
                        trial[dim] += mesh_size[dim] * move
                        # Enforce Bounds
                        if isinstance(bounds[dim][0], float):
                            trial[dim] = float(np.clip(trial[dim], *bounds[dim]))
                        else:
                            trial[dim] = int(np.clip(round(trial[dim]), *bounds[dim]))

                score = self.eval_objfunc(trial, features, target)

                # Check for Improvement
                if (self.opt_type == "max" and score > best_score) or (self.opt_type == "min" and score < best_score):
                    best_score = score
                    best_pos = trial.copy()
                    base_point_pos = trial.copy()
                    improved = True

            # 5.6. Update Mesh Size
            if improved:
                base_point_pos = best_pos.copy()
                for i, b in enumerate(bounds):
                    if not isinstance(b, (list, np.ndarray)):                   # Numerical Only
                        mesh_size[i] *= acc_coeff
                        mesh_size[i] = min(mesh_size[i], (b[1] - b[0]))         # Cap to Parameter Range
            else:
                for i, b in enumerate(bounds):
                    if not isinstance(b, (list, np.ndarray)):                   # Numerical Only
                        mesh_size[i] *= contr_coeff
                        if isinstance(b[0], float):
                            mesh_size[i] = max(mesh_size[i], 1e-6 * (b[1] - b[0]))
                        else:
                            mesh_size[i] = max(1, int(round(mesh_size[i])))

            best_history.append(best_score)

            # 5.7. Early Stopping
            should_stop = True
            for i, b in enumerate(bounds):
                if not isinstance(b, (list, np.ndarray)):
                    param_range = b[1] - b[0]
                    if mesh_size[i] > min_mesh_ratio * param_range:
                        should_stop = False
                        break
            if should_stop:
                print(f"Stopping early: Mesh size < {min_mesh_ratio*100:.1f}% of parameter range.")
                break   
            
            # 5.8. Printing
            if self.verbose == 1:
                print("Iteration #{}".format(iterator))
                print("Corresponding Optimal Solution: {}".format(best_score))
                print("Corresponding Optimum: {}".format(best_pos))
                print("------")
        

        # 6. Storing Optimization Variables in a Dictionary
        self.PS_attritbutes = {
                                "Mesh Size Coefficient": mesh_size_coeff,
                                "Acceleration Coefficient": acc_coeff,
                                "Contraction Coefficient": contr_coeff,
                                "Search Method": search_method
                            }
        

        # 7. Storing Optimization Results in Callable Variables
        self.PS_iterOptSols = {
                                "Iterations' Index": np.arange(1, iterator+1),
                                "Iteration's Optimum Value": best_history
                               }
        self.PS_finalOptimalSol = {
                                    "Optimal Solution Variables' Values": best_pos,
                                    "Optimal Solution Value": best_score
                                    }
        

        # 8. Print a Closure Message
        end_time = time.time()
        total_exec_time = end_time - start_time
        print("Optimization Finished Successfully!")
        print("Optimization Concluded in {} Seconds.".format(np.round(total_exec_time,3)))
        
        return best_history, best_pos, best_score


    # [4] Objective Function Evaluation Function
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