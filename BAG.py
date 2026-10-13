"""
Custom version of BAG file from the Uncertainpy package
"""


import os
import re
import string

from .Arguable import Arguable
from .Argument import Argument
from .Support import Support
from .Attack import Attack


class BAG:

    def __init__(self, path=None):
    
        self.arguments = {}
        self.attacks = []
        self.supports = []

        self.arguable = [] # List of all arguable elements (arguments, attacks, supports)
        
        self.path = path

        if (path is None):
            pass
        else:
            with open(os.path.abspath(path), "r") as f:
                for line in f.readlines():
                    k_name = line.split("(")[0]
                    if k_name in string.whitespace:
                        pass
                    else:
                        k_args = re.findall(rf"{k_name}\((.*?)\)", line)[0].replace(" ", "").split(",")
                        if k_name == "arg":
                            self.add_argument(Argument(k_args[0], float(k_args[1])))
                        elif k_name == "att":
                            attacker = self.arguments[k_args[0]]
                            attacked = self.arguments[k_args[1]]
                            self.add_attack(attacker, attacked)

                        elif k_name == "sup":
                            supporter = self.arguments[k_args[0]]
                            supported = self.arguments[k_args[1]]
                            self.add_support(supporter, supported)

    def add_argument(self, argument, initial_weight=0.5):

        if type(argument) == str:
            argument = Argument(argument, initial_weight)

        if type(argument) != Argument:
            raise TypeError("argument must be of type Argument or str")

        if argument.name in self.arguments:
            raise ValueError(f"Argument with name {argument.name} already exists in BAG")
        
        self.arguments[argument.name] = argument
        self.arguable.append(argument)

        return argument

    def add_attack(self, attacker, attacked, attack_weight=1):

        if type(attacker) != Argument:
            raise TypeError("attacker must be of type Argument")

        if not issubclass(type(attacked), Arguable):
            raise TypeError("attacked must be subclass of Arguable")

        if attacker.name in self.arguments:
            attacker = self.arguments[attacker.name]
        else:
            self.add_argument(attacker)

        if type(attacked) == Argument:
            if attacked.name in self.arguments:
                attacked = self.arguments[attacked.name]
            else:
                self.add_argument(attacked)
                
        new_attack = Attack(attacker, attacked, attack_weight)
        attacked.add_attack(new_attack)
        self.attacks.append(new_attack)
        self.arguable.append(new_attack)

        return new_attack


    def add_support(self, supporter, supported, support_weight=1):
        if type(supporter) != Argument:
            raise TypeError("supporter must be of type Argument")

        if not issubclass(type(supported), Arguable):
            raise TypeError("supported must be subclass of Arguable")

        if supporter.name in self.arguments:
            supporter = self.arguments[supporter.name]
        else:
            self.add_argument(supporter)
        
        if type(supported) == Argument:
            if supported.name in self.arguments:
                supported = self.arguments[supported.name]
            else:
                self.add_argument(supported)

        new_support = Support(supporter, supported, support_weight)
        supported.add_support(new_support)
        self.supports.append(new_support)
        self.arguable.append(new_support)
            
        return new_support

    def remove_attack(self, attack):
        if attack in self.attacks:
            self.attacks.remove(attack)
            self.arguable.remove(attack)
            self.arguments[attack.target.name].attacks.remove(attack)

        else:
            raise ValueError(f"Attack {attack} not found in BAG")
        
    def remove_support(self, support):
        if support in self.supports:
            self.supports.remove(support)
            self.arguable.remove(support)
            self.arguments[support.target.name].supports.remove(support)

        else:
            raise ValueError(f"Support {support} not found in BAG")




    def reset_strength_values(self):
        for a in self.arguable:
            a.strength = a.initial_weight

    def get_arguments(self):
        return list(self.arguments.values())

    def __str__(self) -> str:
        return f"BAG set to read from {self.path} with arguments: {self.arguments}, attacks: {self.attacks} and supports: {self.supports}"

    def __repr__(self) -> str:
        return f"BAG({self.path}) Arguments: {self.arguments} Attacks: {self.attacks} Supports: {self.supports}"

