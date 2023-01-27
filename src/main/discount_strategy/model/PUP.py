from src.main.discount_strategy.model.Vertex import Vertex

class PUP(Vertex):
    def __init__(self,xCoord, yCoord, id, closest_cust_id, number):
        super().__init__(xCoord, yCoord, id)
        self.closest_cust_id = closest_cust_id
        self.number = number

    def __str__(self):
        return ("p " + str(self.id))
