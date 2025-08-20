from typing import Dict ,List ,Optional ,Any 
import networkx as nx 
from pathlib import Path 
import logging 


class RealWorldDataLoader :
    """
    Ładowarka danych z rzeczywistych sieci społecznościowych.
    Obsługuje format SNAP Stanford (Facebook, Google+, Twitter).
    """

    def __init__ (self ,data_dir :str ="data"):
        """
        Args:
            data_dir: Katalog z danymi
        """
        self .data_dir =Path (data_dir )
        self .logger =logging .getLogger (__name__ )

    def load_facebook_ego_network (self ,ego_id :str )->nx .Graph :
        """
        Ładuje ego network z danych Facebook.

        Args:
            ego_id: ID węzła ego (np. "0", "107", "1684")

        Returns:
            Graf reprezentujący ego network
        """
        facebook_dir =self .data_dir /"facebook"


        edges_file =facebook_dir /f"{ego_id }.edges"
        if not edges_file .exists ():
            raise FileNotFoundError (f"Plik edges nie istnieje: {edges_file }")


        graph =nx .Graph ()


        ego_node =int (ego_id )
        graph .add_node (ego_node ,is_ego =True )


        with open (edges_file ,"r")as f :
            for line in f :
                line =line .strip ()
                if line :
                    parts =line .split ()
                    if len (parts )>=2 :
                        node1 ,node2 =int (parts [0 ]),int (parts [1 ])
                        graph .add_edge (node1 ,node2 )


                        graph .add_edge (ego_node ,node1 )
                        graph .add_edge (ego_node ,node2 )


        self ._load_node_features (graph ,facebook_dir ,ego_id )


        self ._load_circles (graph ,facebook_dir ,ego_id )

        self .logger .info (f"Załadowano Facebook ego network {ego_id }: {len (graph .nodes ())} węzłów, {len (graph .edges ())} krawędzi")

        return graph 

    def load_all_facebook_networks (self )->Dict [str ,nx .Graph ]:
        """
        Ładuje wszystkie dostępne ego networks z Facebook.

        Returns:
            Słownik {ego_id: graf}
        """
        facebook_dir =self .data_dir /"facebook"
        networks ={}

        if not facebook_dir .exists ():
            raise FileNotFoundError (f"Katalog Facebook nie istnieje: {facebook_dir }")


        edge_files =list (facebook_dir .glob ("*.edges"))

        for edge_file in edge_files :
            ego_id =edge_file .stem 
            try :
                network =self .load_facebook_ego_network (ego_id )
                networks [ego_id ]=network 
            except Exception as e :
                self .logger .warning (f"Nie udało się załadować network {ego_id }: {e }")

        self .logger .info (f"Załadowano {len (networks )} Facebook ego networks")
        return networks 

    def get_facebook_network_stats (self )->Dict [str ,Dict [str ,Any ]]:
        """
        Pobiera statystyki wszystkich dostępnych Facebook networks.

        Returns:
            Słownik ze statystykami każdego network
        """
        networks =self .load_all_facebook_networks ()
        stats ={}

        for ego_id ,graph in networks .items ():
            stats [ego_id ]={
            "nodes":len (graph .nodes ()),
            "edges":len (graph .edges ()),
            "density":nx .density (graph ),
            "avg_clustering":nx .average_clustering (graph ),
            "is_connected":nx .is_connected (graph ),
            "components":nx .number_connected_components (graph ),
            "avg_degree":sum (dict (graph .degree ()).values ())/len (graph .nodes ())if len (graph .nodes ())>0 else 0 ,
            }


            circles_info =self ._get_circles_info (self .data_dir /"facebook",ego_id )
            if circles_info :
                stats [ego_id ]["circles"]=circles_info 

        return stats 

    def create_combined_facebook_network (self ,max_networks :Optional [int ]=None )->nx .Graph :
        """
        Tworzy połączony graf z wielu ego networks Facebook.

        Args:
            max_networks: Maksymalna liczba networks do połączenia (None = wszystkie)

        Returns:
            Połączony graf
        """
        networks =self .load_all_facebook_networks ()

        if max_networks :

            network_items =list (networks .items ())
            network_items .sort (key =lambda x :len (x [1 ].nodes ()),reverse =True )
            networks =dict (network_items [:max_networks ])

        combined_graph =nx .Graph ()
        node_offset =0 

        for ego_id ,graph in networks .items ():

            mapping ={old_id :old_id +node_offset for old_id in graph .nodes ()}
            shifted_graph =nx .relabel_nodes (graph ,mapping )


            combined_graph =nx .compose (combined_graph ,shifted_graph )


            node_offset +=max (graph .nodes ())+1000 

        self .logger .info (f"Utworzono połączony graf Facebook: {len (combined_graph .nodes ())} węzłów, {len (combined_graph .edges ())} krawędzi")

        return combined_graph 

    def _load_node_features (self ,graph :nx .Graph ,data_dir :Path ,ego_id :str )->None :
        """Wczytuje cechy węzłów z plików .feat i .egofeat."""
        feat_file =data_dir /f"{ego_id }.feat"
        egofeat_file =data_dir /f"{ego_id }.egofeat"
        featnames_file =data_dir /f"{ego_id }.featnames"


        feature_names =[]
        if featnames_file .exists ():
            with open (featnames_file ,"r")as f :
                for line in f :
                    line =line .strip ()
                    if line :
                        parts =line .split (maxsplit =1 )
                        if len (parts )>=2 :
                            feature_names .append (parts [1 ])


        if feat_file .exists ():
            with open (feat_file ,"r")as f :
                for line in f :
                    line =line .strip ()
                    if line :
                        parts =line .split ()
                        if len (parts )>=2 :
                            node_id =int (parts [0 ])
                            features =[int (x )for x in parts [1 :]]

                            if node_id in graph .nodes ():
                                graph .nodes [node_id ]["features"]=features 
                                graph .nodes [node_id ]["feature_count"]=sum (features )


        ego_node =int (ego_id )
        if egofeat_file .exists ()and ego_node in graph .nodes ():
            with open (egofeat_file ,"r")as f :
                line =f .readline ().strip ()
                if line :
                    features =[int (x )for x in line .split ()]
                    graph .nodes [ego_node ]["features"]=features 
                    graph .nodes [ego_node ]["feature_count"]=sum (features )

    def _load_circles (self ,graph :nx .Graph ,data_dir :Path ,ego_id :str )->None :
        """Wczytuje informacje o kręgach społecznych."""
        circles_file =data_dir /f"{ego_id }.circles"

        if not circles_file .exists ():
            return 

        circles =[]
        with open (circles_file ,"r")as f :
            for line in f :
                line =line .strip ()
                if line :
                    parts =line .split ()
                    if len (parts )>=2 :
                        circle_name =parts [0 ]
                        circle_members =[int (x )for x in parts [1 :]if x .isdigit ()]
                        circles .append ({"name":circle_name ,"members":circle_members ,"size":len (circle_members )})


        for node_id in graph .nodes ():
            node_circles =[]
            for i ,circle in enumerate (circles ):
                if node_id in circle ["members"]:
                    node_circles .append (i )
            graph .nodes [node_id ]["circles"]=node_circles 


        graph .graph ["circles"]=circles 

    def _get_circles_info (self ,data_dir :Path ,ego_id :str )->Optional [Dict [str ,Any ]]:
        """Pobiera informacje o kręgach bez ładowania całego grafu."""
        circles_file =data_dir /f"{ego_id }.circles"

        if not circles_file .exists ():
            return None 

        circles =[]
        with open (circles_file ,"r")as f :
            for line in f :
                line =line .strip ()
                if line :
                    parts =line .split ()
                    if len (parts )>=2 :
                        circle_name =parts [0 ]
                        circle_size =len (parts )-1 
                        circles .append ({"name":circle_name ,"size":circle_size })

        return {
        "total_circles":len (circles ),
        "avg_circle_size":sum (c ["size"]for c in circles )/len (circles )if circles else 0 ,
        "max_circle_size":max ((c ["size"]for c in circles ),default =0 ),
        "min_circle_size":min ((c ["size"]for c in circles ),default =0 ),
        }

    def get_suitable_networks_for_testing (self ,min_nodes :int =20 ,max_nodes :int =200 )->List [str ]:
        """
        Znajdź sieci odpowiednie do testowania algorytmów (nie za małe, nie za duże).

        Args:
            min_nodes: Minimalna liczba węzłów
            max_nodes: Maksymalna liczba węzłów

        Returns:
            Lista ID sieci odpowiednich do testowania
        """
        stats =self .get_facebook_network_stats ()
        suitable_networks =[]

        for ego_id ,stat in stats .items ():
            if min_nodes <=stat ["nodes"]<=max_nodes :
                suitable_networks .append (ego_id )


        suitable_networks .sort (key =lambda x :stats [x ]["nodes"])

        return suitable_networks 
