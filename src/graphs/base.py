from abc import ABC ,abstractmethod 
from typing import Any 
import networkx as nx 


class GraphGenerator (ABC ):
    @abstractmethod 
    def generate (self ,n_nodes :int ,**kwargs :Any )->nx .Graph :
        pass 

    @property 
    @abstractmethod 
    def name (self )->str :
        pass 
