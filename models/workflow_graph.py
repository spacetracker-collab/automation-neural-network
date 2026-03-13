import networkx as nx

class WorkflowGraph:

    def __init__(self):

        self.graph = nx.DiGraph()

    def add_bot(self, bot_name):

        self.graph.add_node(bot_name)

    def connect_bots(self, bot1, bot2):

        self.graph.add_edge(bot1, bot2)

    def get_adjacency_matrix(self):

        return nx.adjacency_matrix(self.graph).todense()
