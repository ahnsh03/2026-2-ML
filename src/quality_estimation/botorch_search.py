"""Optional offline BoTorch LogEI baseline over a finite normalized design grid.

References: Snoek et al. (2012); Ament et al. (2023), LogEI; official BoTorch.
The surrogate models experiment losses and is never an online image scorer.
"""
import time
import warnings
import numpy as np


def next_log_expected_improvement(coordinates, indices, losses, initial, seed=281128):
    x=np.asarray(coordinates,float);history=np.asarray(indices,int);loss=np.asarray(losses,float)
    if (x.ndim!=2 or not np.isfinite(x).all() or (x<0).any() or (x>1).any()
            or history.shape!=loss.shape or not np.isfinite(loss).all()
            or len(set(history))!=len(history) or (history<0).any() or (history>=len(x)).any()):
        raise ValueError('invalid LogEI search grid/history')
    for index in initial:
        if not 0<=index<len(x):raise ValueError('invalid initial design index')
        if index not in history:return int(index),dict(stage='common_initial_design')
    remaining=np.array([i for i in range(len(x))if i not in history])
    if not len(remaining):raise ValueError('search space exhausted')
    if len(history)<2:raise ValueError('LogEI requires at least two observations')
    import torch
    import botorch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.analytic import LogExpectedImprovement
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.exceptions.errors import ModelFittingError
    began=time.monotonic()
    active=np.flatnonzero(x.var(0)>1e-14)
    if not len(active):raise ValueError('all BO design coordinates are constant')
    design=x[:,active]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        xx=torch.tensor(design[history],dtype=torch.double);yy=torch.tensor(-loss[:,None],dtype=torch.double)
        # Official SingleTaskGP defaults: dimension-scaled RBF ARD priors and
        # inferred noise. An unregularized custom Matern MLE proved unstable.
        model=SingleTaskGP(xx,yy,outcome_transform=Standardize(m=1))
        failure=None
        with warnings.catch_warnings(record=True)as captured:
            warnings.simplefilter('always')
            try:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model),
                                 optimizer_kwargs={'options':{'maxiter':100}},max_attempts=2)
                acquisition=LogExpectedImprovement(model,best_f=yy.max())
                with torch.no_grad():values=acquisition(torch.tensor(design[remaining,None,:],dtype=torch.double)).cpu().numpy()
                if not np.isfinite(values).all():raise RuntimeError('nonfinite LogEI acquisition')
                selected=int(remaining[np.argmax(values)])
            except (ModelFittingError,RuntimeError)as exc:
                failure=str(exc)
                distance=((design[remaining,None,:]-design[history][None,:,:])**2).sum(2).min(1)
                selected=int(remaining[np.argmax(distance)])
        details=dict(stage='BoTorch_LogEI_default_SingleTaskGP'if failure is None else'logged_space_filling_fallback',version=botorch.__version__,
                     cpu_only=True,seconds=time.monotonic()-began,training_observations=len(history),
                     active_coordinates=active.tolist(),surrogate_failure=failure,
                     selected_log_ei=float(values.max())if failure is None else None,
                     warnings=[str(w.message)for w in captured])
    return selected,details
