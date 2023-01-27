from src.main.discount_strategy.model.Vertex import Vertex

class Customer(Vertex):
    def __init__(self,xCoord, yCoord, id, prob_home, prob_pup, shipping_fee, closest_pup_id):
        super().__init__(xCoord, yCoord, id)
        self.prob_home = prob_home
        self.prob_pup = prob_pup

        self.shipping_fee = shipping_fee
        self.closest_pup_id = closest_pup_id
    def __str__(self):
        return (str(self.id))

id = 15
scenario = 12

print("id",  bin(~id &(2**4- 1)) ,"scenario", bin(scenario), ~id &(2**4- 1)& scenario, bin((~id &(2**4- 1))& scenario))
if not (~id &(2**4- 1))& scenario:
    print("here")

