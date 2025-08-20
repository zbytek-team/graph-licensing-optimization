import random 
import math 
from typing import Any ,List ,Optional 
import networkx as nx 
from src .core import Solution ,LicenseGroup ,LicenseType ,Algorithm 
from src .core import SolutionValidator 
from src .utils import SolutionBuilder 


class SimulatedAnnealing (Algorithm ):
    @property 
    def name (self )->str :
        return "simulated_annealing"

    def __init__ (
    self ,
    initial_temperature :float =100.0 ,
    cooling_rate :float =0.995 ,
    min_temperature :float =0.001 ,
    max_iterations :int =20000 ,
    max_iterations_without_improvement :int =2000 ,
    ):
        self .initial_temperature =initial_temperature 
        self .cooling_rate =cooling_rate 
        self .min_temperature =min_temperature 
        self .max_iterations =max_iterations 
        self .max_iterations_without_improvement =max_iterations_without_improvement 
        self .validator =SolutionValidator ()

    def solve (self ,graph :nx .Graph ,license_types :List [LicenseType ],**kwargs :Any )->Solution :
        current_solution =self ._generate_random_initial_solution (graph ,license_types )
        if not self .validator .is_valid_solution (current_solution ,graph ):
            from .greedy import GreedyAlgorithm 

            current_solution =GreedyAlgorithm ().solve (graph ,license_types )
            if not self .validator .is_valid_solution (current_solution ,graph ):
                return Solution ([],0.0 ,set ())

        best_solution =current_solution 
        temperature =self .initial_temperature 
        iterations_without_improvement =0 

        for i in range (self .max_iterations ):
            if temperature <self .min_temperature :
                break 

            neighbor =self ._generate_neighbor (current_solution ,graph ,license_types )

            if neighbor and self .validator .is_valid_solution (neighbor ,graph ):
                delta_cost =neighbor .total_cost -current_solution .total_cost 
                if delta_cost <0 or random .random ()<math .exp (-delta_cost /max (temperature ,1e-12 )):
                    current_solution =neighbor 
                    if current_solution .total_cost <best_solution .total_cost :
                        best_solution =current_solution 
                        iterations_without_improvement =0 
                    else :
                        iterations_without_improvement +=1 
                else :
                    iterations_without_improvement +=1 
            else :
                iterations_without_improvement +=1 

            if iterations_without_improvement >=self .max_iterations_without_improvement :
                iterations_without_improvement =0 
                temperature =max (self .min_temperature ,temperature *0.5 )

            temperature *=self .cooling_rate 

        return best_solution 

    def _generate_random_initial_solution (self ,graph :nx .Graph ,license_types :List [LicenseType ])->Solution :
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
                license_type =random .choice (compatible )
                members =set (random .sample (list (available ),min (len (available ),license_type .max_capacity )))
            else :
                license_type =SolutionBuilder .find_cheapest_single_license (license_types )
                members ={owner }

            additional =members -{owner }
            groups .append (LicenseGroup (license_type ,owner ,additional ))
            uncovered -=members 

        return SolutionBuilder .create_solution_from_groups (groups )

    def _generate_neighbor (self ,solution :Solution ,graph :nx .Graph ,license_types :List [LicenseType ])->Optional [Solution ]:
        strategies =[
        self ._change_license_type ,
        self ._move_node ,
        self ._swap_nodes ,
        self ._merge_groups ,
        self ._split_group ,
        ]

        for _ in range (12 ):
            strategy =random .choice (strategies )
            try :
                candidate =strategy (solution ,graph ,license_types )
            except Exception :
                candidate =None 
            if candidate and self .validator .is_valid_solution (candidate ,graph ):
                return candidate 
        return None 

    def _change_license_type (self ,solution :Solution ,graph :nx .Graph ,license_types :List [LicenseType ])->Optional [Solution ]:
        group =random .choice (solution .groups )
        new_license =SolutionBuilder .find_cheapest_license_for_size (group .size ,license_types )
        if not new_license or new_license ==group .license_type :
            return None 

        new_groups =[g for g in solution .groups if g is not group ]
        new_groups .append (LicenseGroup (new_license ,group .owner ,set (group .additional_members )))
        return SolutionBuilder .create_solution_from_groups (new_groups )

    def _move_node (self ,solution :Solution ,graph :nx .Graph ,license_types :List [LicenseType ])->Optional [Solution ]:
        source_candidates =[g for g in solution .groups if g .size >g .license_type .min_capacity ]
        if not source_candidates :
            return None 
        source =random .choice (source_candidates )
        node =random .choice (list (source .all_members ))

        target_candidates =[g for g in solution .groups if g is not source and g .size <g .license_type .max_capacity ]
        if not target_candidates :
            return None 
        target =random .choice (target_candidates )

        new_src_members =set (source .all_members )-{node }
        new_tgt_members =set (target .all_members )|{node }

        if not new_src_members or not new_tgt_members :
            return None 

        if not nx .is_connected (graph .subgraph (new_src_members ))or not nx .is_connected (graph .subgraph (new_tgt_members )):
            return None 

        new_src_license =SolutionBuilder .find_cheapest_license_for_size (len (new_src_members ),license_types )
        new_tgt_license =SolutionBuilder .find_cheapest_license_for_size (len (new_tgt_members ),license_types )
        if not new_src_license or not new_tgt_license :
            return None 

        new_groups =[g for g in solution .groups if g not in (source ,target )]
        owner_src =next (iter (new_src_members ))
        owner_tgt =next (iter (new_tgt_members ))
        new_groups .append (LicenseGroup (new_src_license ,owner_src ,new_src_members -{owner_src }))
        new_groups .append (LicenseGroup (new_tgt_license ,owner_tgt ,new_tgt_members -{owner_tgt }))
        return SolutionBuilder .create_solution_from_groups (new_groups )

    def _swap_nodes (self ,solution :Solution ,graph :nx .Graph ,license_types :List [LicenseType ])->Optional [Solution ]:
        if len (solution .groups )<2 :
            return None 
        g1 ,g2 =random .sample (solution .groups ,2 )
        n1 =random .choice (list (g1 .all_members ))
        n2 =random .choice (list (g2 .all_members ))

        new_g1 =(set (g1 .all_members )-{n1 })|{n2 }
        new_g2 =(set (g2 .all_members )-{n2 })|{n1 }

        if not nx .is_connected (graph .subgraph (new_g1 ))or not nx .is_connected (graph .subgraph (new_g2 )):
            return None 

        l1 =SolutionBuilder .find_cheapest_license_for_size (len (new_g1 ),license_types )
        l2 =SolutionBuilder .find_cheapest_license_for_size (len (new_g2 ),license_types )
        if not l1 or not l2 :
            return None 

        new_groups =[g for g in solution .groups if g not in (g1 ,g2 )]
        o1 =next (iter (new_g1 ))
        o2 =next (iter (new_g2 ))
        new_groups .append (LicenseGroup (l1 ,o1 ,new_g1 -{o1 }))
        new_groups .append (LicenseGroup (l2 ,o2 ,new_g2 -{o2 }))
        return SolutionBuilder .create_solution_from_groups (new_groups )

    def _merge_groups (self ,solution :Solution ,graph :nx .Graph ,license_types :List [LicenseType ])->Optional [Solution ]:
        if len (solution .groups )<2 :
            return None 
        g1 ,g2 =random .sample (solution .groups ,2 )
        merged =set (g1 .all_members )|set (g2 .all_members )
        if not nx .is_connected (graph .subgraph (merged )):
            return None 
        lg =SolutionBuilder .merge_groups (g1 ,g2 ,graph ,license_types )
        if not lg :
            return None 
        new_groups =[g for g in solution .groups if g not in (g1 ,g2 )]
        new_groups .append (lg )
        return SolutionBuilder .create_solution_from_groups (new_groups )

    def _split_group (self ,solution :Solution ,graph :nx .Graph ,license_types :List [LicenseType ])->Optional [Solution ]:
        large =[g for g in solution .groups if g .size >=3 ]
        if not large :
            return None 
        group =random .choice (large )
        nodes =list (group .all_members )
        sub =graph .subgraph (nodes )
        if not nx .is_connected (sub ):
            return None 
        cut =nx .minimum_edge_cut (sub )
        if not cut :
            return None 
        Gp =sub .copy ()
        Gp .remove_edges_from (cut )
        components =list (nx .connected_components (Gp ))
        if len (components )<2 :
            return None 
        new_groups :List [LicenseGroup ]=[g for g in solution .groups if g !=group ]
        for comp in components :
            lt =SolutionBuilder .find_cheapest_license_for_size (len (comp ),license_types )
            if not lt :
                return None 
            owner =next (iter (comp ))
            new_groups .append (LicenseGroup (lt ,owner ,set (comp )-{owner }))
        return SolutionBuilder .create_solution_from_groups (new_groups )
