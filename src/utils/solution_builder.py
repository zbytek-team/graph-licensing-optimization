from typing import List ,Set ,Optional 
import networkx as nx 
from src .core import Solution ,LicenseGroup ,LicenseType 


class SolutionBuilder :
    @staticmethod 
    def create_solution_from_groups (groups :List [LicenseGroup ])->Solution :
        total_cost =sum (group .license_type .cost for group in groups )
        covered_nodes =set ()
        for group in groups :
            covered_nodes .update (group .all_members )
        return Solution (groups =groups ,total_cost =total_cost ,covered_nodes =covered_nodes )

    @staticmethod 
    def get_compatible_license_types (group_size :int ,license_types :List [LicenseType ],exclude :Optional [LicenseType ]=None )->List [LicenseType ]:
        compatible =[]
        for license_type in license_types :
            if exclude and license_type ==exclude :
                continue 
            if license_type .min_capacity <=group_size <=license_type .max_capacity :
                compatible .append (license_type )
        return compatible 

    @staticmethod 
    def get_owner_neighbors_with_self (graph :nx .Graph ,owner :int )->Set [int ]:
        return set (graph .neighbors (owner ))|{owner }

    @staticmethod 
    def merge_groups (group1 :LicenseGroup ,group2 :LicenseGroup ,graph :nx .Graph ,license_types :List [LicenseType ])->Optional [LicenseGroup ]:
        combined_members =group1 .all_members |group2 .all_members 
        combined_size =len (combined_members )

        for license_type in license_types :
            if license_type .min_capacity <=combined_size <=license_type .max_capacity :
                for potential_owner in combined_members :
                    owner_neighbors =SolutionBuilder .get_owner_neighbors_with_self (graph ,potential_owner )
                    if combined_members .issubset (owner_neighbors ):
                        additional_members =combined_members -{potential_owner }
                        return LicenseGroup (license_type ,potential_owner ,additional_members )
        return None 

    @staticmethod 
    def find_cheapest_single_license (license_types :List [LicenseType ])->LicenseType :
        single_licenses =[lt for lt in license_types if lt .min_capacity <=1 ]
        if not single_licenses :
            return min (license_types ,key =lambda lt :lt .cost )
        return min (single_licenses ,key =lambda lt :lt .cost )

    @staticmethod 
    def find_cheapest_license_for_size (size :int ,license_types :List [LicenseType ])->Optional [LicenseType ]:
        compatible =[lt for lt in license_types if lt .min_capacity <=size <=lt .max_capacity ]
        if not compatible :
            return None 
        return min (compatible ,key =lambda lt :lt .cost )

    @staticmethod 
    def build_solution_for_subset (nodes :List ,graph :nx .Graph ,license_types :List [LicenseType ])->Solution :
        from src .algorithms .greedy import GreedyAlgorithm 

        subgraph =graph .subgraph (nodes )
        greedy_solver =GreedyAlgorithm ()
        return greedy_solver .solve (subgraph ,license_types )
