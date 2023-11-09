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
        """
        The constructor for Visualizer class.

        Initializes the plotter, tinker gui, actors, and meshes list.
        """
        # Create an Optimizer
        self.optimizer = None
        
        # Create a Vedo Plotter
        self.plotter = vd.Plotter(axes=1)
        
        # Create a Tkinter root window for the file dialog
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window
        self.opt_menu = None # Menu window

        self.constraints = None # Constraints dictionary {name: constraint}
        self.weights = None # List of weights
        self.iterations = 1 # Number of iterations

        self.meshes = [] # List of meshes
        self.actors = {} # Dictionary of actors
        self.initialized = False # Flag to check if the geometry is initialized

    #-------------------------MENU METHODS---------------------------#

    def init_Visualizer(self) -> None:
        """
        Initializes the menu and constraints for the visualizer
        """
        self._init_menu()


    def init_menu(self) -> None:
        """ 
        Here we initialize the menu as convenient
        """
        pass

    def _init_menu(self) -> None:
        """ 
        Setup the basic initial menu buttons in Plotter
        """

        # Load mesh button
        self.plotter.add_button(self.load_mesh, 
                                           pos=(0.9, 0.9), 
                                           states=["Load Mesh"],
                                           c="w",
                                           bc="k",
                                           font="Calco",
                                           )

        # Optimizaiton button
        self.plotter.add_button(self.optimizations_settings, 
                                           pos=(0.2, 0.9), 
                                           states=["Optimization Settings"],
                                           c="w",
                                           bc="k",
                                           font="Calco",
                                           )
        
        # Reset button
        self.plotter.add_button(self.reset_scene,
                                             pos=(0.1, 0.85), 
                                             states=["Reset"],
                                             c="w",
                                             bc="k",
                                             font="Calco",
                                             )


        self.init_menu()
        
        # # Show main plotter
        self.plotter.show(interactive=True, axes=1 )
    
    def display_weights(self) -> None:
        """ 
        Function that displays the weights in plot window
        """

        for i, (name, value) in enumerate(self.weights.items()):
            t_w = vd.Text2D(f"Weight {name}: {value}", 
                                  pos=(0.04, 0.75-i*0.05),
                                  font="Calco",
                                  s=1.1,
                                  c="k",
                                  )
            self.add_to_scene("weight_"+name, t_w)
        self.plotter.render()


    #-------------------------OPTIMIZATION METHODS---------------------------#

    def setup(self)-> None:
        """ 
        Here we setup the geometry we are going to optimize
        """
        pass


    def optimization_run(self) -> None:
        """ 
        Function that runs the optimization
        """

        # Set weights for the constraints
        for name, constraint in self.constraints.items():
                constraint.set_weigth(self.weights[name])

        for _ in range(self.iterations):
            
            # Per constraint compute the gradients
            for name, constraint in self.constraints.items():                
                self.optimizer.get_gradients(constraint)
            
            # Optimize step
            self.optimizer.optimize()

            # Update the scene
            self.update_scene(self.optimizer.X)



    def optimizations_settings(self) -> None:
        """ 
        Function that sets up the optimization settings menu
        """
        # setup the geometry
        
        if not self.initialized:
            self.setup()
            self.initialized = True

        # Function to retrieve and display the entered values
        def run_opt():  
            # Get the values from the Entry widgets and convert them to float
            entered_values = [float(entry.get()) for entry in entry_widgets]
            self.weights = dict(zip(self.weights.keys(), entered_values[:-1]))
            self.iterations = int(entered_values[-1])
            # Display the values in plotter
            self.display_weights()    
            
            self.opt_menu.quit()
            self.opt_menu.destroy()

            # Run optimization
            self.optimization_run()

        self.opt_menu = Toplevel(self.root)
        self.opt_menu.title("Optimization Settings")


        # Create Entry widgets and labels dynamically
        entry_widgets = []
        for i, (name, value) in enumerate(self.weights.items()):
            
            label = Label(self.opt_menu, text=f"Weight {name}:")
            entry = Entry(self.opt_menu)
            entry.insert(0, str(value))  # Set default value in the Entry widget
            label.grid(row=i, column=0)
            entry.grid(row=i, column=1)
            entry_widgets.append(entry)

        # Define iterations entry
        iterations_label = Label(self.opt_menu, text=f"Iterations :")
        iterations_entry = Entry(self.opt_menu)
        iterations_entry.insert(0, str(self.iterations))  # Set default value in the Entry widget
        iterations_label.grid(row=len(self.weights), column=0)
        iterations_entry.grid(row=len(self.weights), column=1)
        entry_widgets.append(iterations_entry)

        # Create a button to trigger the input retrieval
        button_show = Button(self.opt_menu, text="Optimize", command=run_opt)

        # Create a label to display the result
        label_result = Label(self.opt_menu, text="")

        # Arrange widgets using the grid layout
        button_show.grid (row=len(self.weights) + 1, columnspan=2)
        label_result.grid(row=len(self.weights) + 2, columnspan=2)

        self.root.mainloop()

    #-------------------------UPDATE SCENE METHODS---------------------#

    def reset_scene(self) -> None:
        """ 
        Function that resets the scene
        """
        self.optimizer.reset()
        self.update_scene(self.optimizer.X)
        pass


    def update_scene(self, X):
        """ 
        Function that updates the scene, i.e. modify the meshes according to the variables X

        Args:
            X (numpy array): array of variables
        """
        pass


    #-------------------------MESH METHODS---------------------------#

    def load_mesh(self) -> None:
        """ 
        Function that loads a mesh
        """
        file_dialog_options = {
        "title": "Select a File",
        "filetypes": [("Wave Font", "*.obj"), ("All Files", "*.*")]
        }
        file_path = filedialog.askopenfilename(**file_dialog_options)
        
        if file_path:
            actor = vd.load(file_path)
            self.meshes.append( [actor.points(), actor.faces()])
            self.show_meshes()
            #self.plotter.add(actor)

        self.plotter.render()

        
    def show_meshes(self) -> None:
        """ 
        Function that shows the meshes
        """
        self.plotter.clear()
        for i, mesh in enumerate(self.meshes):
            actor = vd.Mesh([mesh[0], mesh[1]])
            actor.lc("k").lw(0.1).c("r").alpha(0.5)
            self.add_to_scene("mesh_"+str(i), actor)
        
        self.plotter.render()

  
    

    #-----------------------ADD/REMOVE METHODS-----------------------#

    def remove_from_scene(self, name):
        """ 
        Remove an actor from the scene

        Args:
            name (str): name of the actor
        """
        if name in self.actors:
            self.plotter.remove(self.actors[name])
        else:
            print(f"Actor {name} not in scene")
            return
        
        

    def add_to_scene(self, name, act):
        """ 
        Add an actor to the scene

        Args:
            name (str): name of the actor
            act (actor): actor to be added
        """
        if name in self.actors:
            self.remove_from_scene(name)
        self.actors[name] = act 
        self.plotter.add(act)

