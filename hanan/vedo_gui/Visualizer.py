import os 
import sys

# Add hananLab to path
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

import vedo as vd 
import tkinter as tk
from tkinter import Entry, Label, Button, filedialog, Toplevel
import numpy as np
from hanan.optimization.Optimizer import Optimizer


class Visualizer():

    def __init__(self) -> None:
        # Create an Optimizer
        self.optimizer = Optimizer()
        
        # Create a Vedo Plotter
        self.plotter = vd.Plotter(axes=1)
        
        # Create a Tkinter root window for the file dialog
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window

        self.constraints = None # Constraints dictionary {name: constraint}
        self.weights = None # Weights dictionary {name: weight}

        self.meshes = [] # List of meshes

    def setup(self)-> None:
        """ Here we setup the geometry we are goint to optimize
        """

    def init_Visualizer(self, constraints, weights) -> None:
        self.constraints = constraints
        self.weights = weights
        self._init_menu()

    def init_menu(self) -> None:
        """ Here we initialize the menu as convenient
        """
        pass

    def _init_menu(self) -> None:
        """ Here we setup the basic initial menu
        """

        self.plotter.add_button(self.load_mesh, 
                                           pos=(0.9, 0.9), 
                                           states=["Load Mesh"],
                                           c="w",
                                           bc="k",
                                           font="Calco",
                                           )

        
        self.plotter.add_button(self.optimizations_settings, 
                                           pos=(0.2, 0.9), 
                                           states=["Optimization Settings"],
                                           c="w",
                                           bc="k",
                                           font="Calco",
                                           )
        

        self.init_menu()
        
        # # Show main plotter
        self.plotter.show(interactive=True)

    

    def optimizations_settings(self) -> None:
              
        # Function to retrieve and display the entered values
        def show_values():
            try:
                # Get the values from the Entry widgets and convert them to float
                entered_values = [float(entry.get()) for entry in entry_widgets]

                # Display the values
                label_result.config(text="Entered Values: " + ", ".join(map(str, entered_values)))
            except ValueError:
                # Handle the case where the input is not a valid float
                label_result.config(text="Invalid input. Please enter valid numbers.")

            self.opt_menu.quit()
            self.opt_menu.destroy()

        self.opt_menu = Toplevel(self.root)
        self.opt_menu.title("Optimization Settings")

        # Define widget names and values
        W = ["w1", "w2", "w3"]
        V = [0.0, 0.0, 0.0]  # Initialize with default values

        # Create Entry widgets and labels dynamically
        entry_widgets = []
        for i, widget_name in enumerate(W):
            label = Label(self.opt_menu, text=f"Enter {widget_name}:")
            entry = Entry(self.opt_menu)
            entry.insert(0, str(V[i]))  # Set default value in the Entry widget
            label.grid(row=i, column=0)
            entry.grid(row=i, column=1)
            entry_widgets.append(entry)

        # Create a button to trigger the input retrieval
        button_show = Button(self.opt_menu, text="Show Values", command=show_values)

        # Create a label to display the result
        label_result = Label(self.opt_menu, text="")

        # Arrange widgets using the grid layout
        button_show.grid(row=len(W), columnspan=2)
        label_result.grid(row=len(W) + 1, columnspan=2)

        self.root.mainloop()


    def show_meshes(self) -> None:
        """ Show the meshes
        """
        self.plotter.clear()
        for mesh in self.meshes:
            actor = vd.Mesh([mesh[0], mesh[1]])
            self.plotter.add(actor)
        

    def load_mesh(self) -> None:
        """ Load a mesh
        """
        file_dialog_options = {
        "title": "Select a File",
        "filetypes": [("Wave Font", "*.obj"), ("All Files", "*.*")]
        }
        file_path = filedialog.askopenfilename(**file_dialog_options)
        
        if file_path:
            actor = vd.load(file_path)
            self.meshes.append( [actor.points(), actor.faces()])
            self.plotter.add(actor)
        #self.show_meshes()
 
