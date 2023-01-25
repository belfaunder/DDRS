from Vertex import Vertex

class Customer(Vertex):
    def __init__(self,xCoord, yCoord, id, prob_home, prob_pup, shipping_fee):
        super().__init__(xCoord, yCoord, id)
        self.prob_home = prob_home
        self.prob_pup = prob_pup

        self.shipping_fee = shipping_fee

    def __str__(self):
        return (str(self.id))

