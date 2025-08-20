from src .core import LicenseType ,Solution ,Algorithm ,LicenseGroup 
from typing import Any ,List 
import networkx as nx 
import pulp 


class ILPSolver (Algorithm ):
    @property 
    def name (self )->str :
        return "ilp"

    def solve (self ,graph :nx .Graph ,license_types :List [LicenseType ],**kwargs :Any )->Solution :
        nodes =list (graph .nodes ())
        prob =pulp .LpProblem ("Graph_Licensing_Optimization",pulp .LpMinimize )
        x ={}
        for i in nodes :
            neighbors_i =set (graph .neighbors (i ))|{i }
            for j in neighbors_i :
                for t_idx ,license_type in enumerate (license_types ):
                    x [i ,j ,t_idx ]=pulp .LpVariable (f"x_{i }_{j }_{t_idx }",cat ="Binary")
        group_active ={}
        for i in nodes :
            for t_idx ,license_type in enumerate (license_types ):
                group_active [i ,t_idx ]=pulp .LpVariable (f"group_active_{i }_{t_idx }",cat ="Binary")
        total_cost =pulp .lpSum ([group_active [i ,t_idx ]*license_type .cost for i in nodes for t_idx ,license_type in enumerate (license_types )])
        prob +=total_cost 
        for j in nodes :
            prob +=pulp .lpSum ([x .get ((i ,j ,t_idx ),0 )for i in nodes for t_idx in range (len (license_types ))])==1 
        for i in nodes :
            for t_idx ,license_type in enumerate (license_types ):
                group_size =pulp .lpSum ([x .get ((i ,j ,t_idx ),0 )for j in nodes ])
                prob +=group_size <=group_active [i ,t_idx ]*license_type .max_capacity 
                prob +=group_size >=group_active [i ,t_idx ]*license_type .min_capacity 
        for i in nodes :
            for t_idx ,license_type in enumerate (license_types ):
                if (i ,i ,t_idx )in x :
                    prob +=x [i ,i ,t_idx ]>=group_active [i ,t_idx ]
        prob .solve (pulp .PULP_CBC_CMD (msg =0 ))
        if prob .status !=pulp .LpStatusOptimal :
            raise RuntimeError (f"ILP solver failed with status: {pulp .LpStatus [prob .status ]}")
        groups =[]
        covered_nodes =set ()
        for i in nodes :
            for t_idx ,license_type in enumerate (license_types ):
                if group_active [i ,t_idx ].varValue and group_active [i ,t_idx ].varValue >0.5 :
                    group_members =set ()
                    for j in nodes :
                        if (i ,j ,t_idx )in x and x [i ,j ,t_idx ].varValue and x [i ,j ,t_idx ].varValue >0.5 :
                            group_members .add (j )
                    if group_members :
                        additional_members =group_members -{i }
                        group =LicenseGroup (
                        license_type =license_type ,
                        owner =i ,
                        additional_members =additional_members ,
                        )
                        groups .append (group )
                        covered_nodes .update (group_members )
        total_cost_value =sum (group .license_type .cost for group in groups )
        return Solution (groups =groups ,total_cost =total_cost_value ,covered_nodes =covered_nodes )
