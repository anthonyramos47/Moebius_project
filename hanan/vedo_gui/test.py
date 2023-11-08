import tkinter as tk
from vedo import Plotter, show, Text2D

class YourApp:
    def __init__(self):
        # Create the main tkinter window
        self.root = tk.Tk()
        self.root.title("Main Window")

        # Create a Vedo plotter
        self.plotter = Plotter(axes=1, offscreen=True)

        # Insert the Vedo plotter into the Tkinter window
        self.plotter.renderers[0].window = self.root

        # Add a button to the Tkinter window to start the Vedo loop
        self.start_button = tk.Button(self.root, text="Start Vedo Loop", command=self.start_vedo_loop)
        self.start_button.pack()

    def start_vedo_loop(self):
        # Start the Vedo loop
        #self.root.after(10, self.update_vedo)  # Update Vedo every 10 ms
        self.plotter.show(interactive=True)

    def update_vedo(self):
        # Update the Vedo plotter
        #self.plotter.renderers[0].update()

        # Continue the Vedo loop if needed
        if self.plotter.interactive:
            self.root.after(10, self.update_vedo)

if __name__ == "__main__":
    app = YourApp()
    app.root.mainloop()
