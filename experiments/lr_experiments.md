# Logistic Regression Experiments
## Experiments 1: Small Learning Rate

**Setting**
- learning Rate: 0.0001
- epochs: 1000

**Observation**
- loss decreases slowly
- weight change gradually
- training is stable

**Conclusion**
A small learning rate leads to stable but slow convergence

---

## Experiments 2: Larger Learning Rate

**Setting**
- learning Rate: 0.01
- epochs: 1000

**Observation**
- loss decreases faster
- weight update more aggressively

**Conclusion**
A large learning rate **speeds up** training but may become unstale if too large.

---

## Experiments 3: Very Large Learning Rate

**Setting**
- learning Rate: 1.0
- epochs: 1000

**Observation**
- predictions become extreme (close to 0 or 1)
- weight grow very large
- model become unstable

**Conclusion**
Too large learning rate causes divergence.

---

## Experiments 4: Feature Scaling

**Setting**
- with scaling vs without scaling

**Observation**
- with scaling: traning becomes more balanced
- without scaling: one feature dominates updates

**Conclusion**
Feature scaling is important in multi-feature modles.


Update direction is determined by error
Update magnitude is determined by X