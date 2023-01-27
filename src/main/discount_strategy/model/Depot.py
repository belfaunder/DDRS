from src.main.discount_strategy.model.Vertex import Vertex

class Depot(Vertex):
    def __str__(self):
        return ("d "+str(self.id))
