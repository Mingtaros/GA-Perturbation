import os
import sys
import math
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.decomposition.asf import ASF
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from torch.utils.data import DataLoader

from perturbation import perturb
from model import load_mnist_dataset, load_model, test_model


class MinimumPerturbationProblem(ElementwiseProblem):
    def __init__(self, timestamp, image_shape, norm='l1_norm'):
        self.timestamp = timestamp # already in string
        possible_norms = ['l1_norm', 'l2_norm', 'linfinity_norm']
        if norm not in possible_norms:
            ValueError(f"Norm {norm} Invalid. Choose from: {', '.join(possible_norms)}.")
        self.norm = norm

        self.x_shape = image_shape

        # for testing
        self.device = torch.device('mps')

        super().__init__(
            n_var=math.prod(self.x_shape),
            n_obj=2, # minimize perturbation, minimize model accuracy
            xl=0,
            xu=1,
        )


    def _evaluate(self, x, out, *args, **kwargs):
        perturbation = x.reshape(self.x_shape)

        out["F"] = []

        if self.norm == "l1_norm":
            norm_value = np.sum(np.abs(perturbation))
        elif self.norm == "l2_norm":
            norm_value = np.sqrt(np.sum(np.square(perturbation)))
        else:
            # self.norm == "linfinity_norm"
            norm_value = np.max(np.abs(perturbation))
        out["F"].append(norm_value)

        _, test_dataset = load_mnist_dataset()
        perturbed_test_dataset = perturb(test_dataset, perturbation)
        perturbed_test_loader = DataLoader(dataset=perturbed_test_dataset, batch_size=1000, shuffle=False)

        model = load_model(f"models/model_{self.timestamp}.pth", device=self.device) # for now hardcode
        test_accuracy = test_model(model, perturbed_test_loader, self.device)

        out["F"].append(test_accuracy)


def main(num_generations=10, norm="l2_norm", priority_weights=[0.5, 0.5]):
    torch.manual_seed(42)
    timestamp = datetime.datetime.today().strftime('%Y%m%d')
    problem = MinimumPerturbationProblem(
        timestamp=timestamp,
        image_shape=(1, 28, 28),
        norm=norm
    )
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=UniformCrossover(),
        mutation=PolynomialMutation(),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", num_generations)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        save_history=True,
        verbose=True
    )

    # choose one between the rest
    approx_ideal = res.F.min(axis=0)
    approx_nadir = res.F.max(axis=0)
    # minmax scaler
    nF = (res.F - approx_ideal) / (approx_nadir - approx_ideal)

    weights = np.array(priority_weights)
    decomp = ASF()
    # best result according to weights
    i = decomp.do(nF, 1/weights).argmin()

    print("Best solution found according to weights: \nX = %s\nF = %s" % (res.X[i], res.F[i]))

    # save the best res.X
    reshaped_image = res.X[i].reshape(problem.x_shape)
    if not os.path.exists('images'):
        os.makedirs('images')
    np.save(f'images/perturbed_image_{timestamp}.npy', reshaped_image)

    _, test_dataset = load_mnist_dataset()
    # get random image
    original_image, _ = test_dataset[np.random.randint(0, len(test_dataset))]
    perturbed_image = reshaped_image.reshape(28, 28)
    # plot the original and perturbed images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original_image.squeeze(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(perturbed_image, cmap='gray')
    axes[1].set_title(f'Perturbed Image, Norm={res.F[i, 0]:.2f}, Accuracy={res.F[i, 1]:.2f}%')
    axes[1].axis('off')
    
    fig.savefig(f'images/original_vs_perturbed_image_{timestamp}.png')
    plt.show()


if __name__ == "__main__":
    try:
        num_generations = int(sys.argv[1])
        norm = sys.argv[2]
    except (IndexError, ValueError):
        print("Usage: python genetic.py <num_generations> <norm>")
        sys.exit(1)
    priority_weights = [0.5, 0.5]
    main(num_generations, norm, priority_weights)
