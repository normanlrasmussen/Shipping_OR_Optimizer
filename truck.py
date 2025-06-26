# This will contain the truck class

class Truck:
    def __init__(self, name:str, capacity:int):
        self.name = name
        self.capacity = capacity
        self.items = None
    
    def __str__(self):
        return f"{self.name} has {self.capacity} remaining space."
    
    def add_item(self, name:str, quantity:int, size_per_item:int):
        # add singular item
        
        # verify item can be added
        if self.capacity < quantity * size_per_item:
            raise ValueError(f"Not enough space for {name} to be added")
        
        if self.items == None:
            self.items = []
        self.items.append(
            {
                name : {
                    "quantity" : quantity,
                    "size_per_item" : size_per_item
                }

             }
        )

        # Update space
        self.capacity -= quantity * size_per_item
    
    def add_items(self, items:list):
        # Add a list of items of the form [[name, quantity, size_per_item], ...]
        for name, quantity, size_per_item in items:
            self.add_item(name, quantity, size_per_item)

