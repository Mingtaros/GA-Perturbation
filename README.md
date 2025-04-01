# GA-Perturbation
Some time ago I discussed with my friend jokingly about GA might be good to make minimum perturbation for machine learning models. This might be true, might be false. Working on it might trigger some PTSD we have from what's supposed to be a philosophical AI mod. Unironically, I'm now curious whether this might work or not.

## What is it?
This is a small experiment on how good is Genetic Algorithm in creating a minimum perturbation for machine learning models in order to throw off their predictions (doesn't matter to where). The goal is to minimize both perturbation and model accuracy.

## How to Run?
1. Install necessary requirements
2. Train the model, run `python model.py` in terminal. Make sure folder `models` is created with a model.
3. Run Genetic Algorithm, `python genetic.py <num_generations>` in terminal for the main result.
    - use any number for `<num_generations>`, by default, it's 10. The more generation, the better the result would be.
    - note that this script uses the timestamp of the current day to both find the model and save images
3. See result in images for the perturbation image.
