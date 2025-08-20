from typing import List 
import random 
import networkx as nx 
from src .core import LicenseType ,Solution ,LicenseGroup 
from src .core import SolutionValidator 
from src .utils import SolutionBuilder 


class PopulationManager :
    def __init__ (self ,validator :SolutionValidator ):
        self .validator =validator 

    def initialize_population (self ,graph :nx .Graph ,license_types :List [LicenseType ],population_size :int )->List [Solution ]:
        population :List [Solution ]=[]
        attempts =0 
        while len (population )<population_size and attempts <population_size *5 :
            attempts +=1 
            sol =self .generate_truly_random_solution (graph ,license_types )
            if sol and self .validator .is_valid_solution (sol ,graph ):
                population .append (sol )


        if len (population )<population_size :
            from ..greedy import GreedyAlgorithm 

            greedy =GreedyAlgorithm ()
            greedy_sol =greedy .solve (graph ,license_types )
            while len (population )<population_size :
                population .append (greedy_sol )

        return population [:population_size ]

    def generate_truly_random_solution (self ,graph :nx .Graph ,license_types :List [LicenseType ])->Solution :
        nodes =list (graph .nodes ())
        random .shuffle (nodes )
        uncovered =set (nodes )
        groups :List [LicenseGroup ]=[]

        while uncovered :
            owner =random .choice (list (uncovered ))
            neighbors =SolutionBuilder .get_owner_neighbors_with_self (graph ,owner )&uncovered 
            available =neighbors |{owner }

            compatible =[lt for lt in license_types if lt .min_capacity <=len (available )<=lt .max_capacity ]
            if compatible :
                lt =random .choice (compatible )
                members =set (random .sample (list (available ),min (len (available ),lt .max_capacity )))
            else :
                lt =SolutionBuilder .find_cheapest_single_license (license_types )
                members ={owner }

            groups .append (LicenseGroup (lt ,owner ,members -{owner }))
            uncovered -=members 

        return SolutionBuilder .create_solution_from_groups (groups )
