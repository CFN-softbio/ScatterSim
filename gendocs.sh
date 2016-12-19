#!/bin/bash
pydoc -w ScatterSim.BaseClasses
pydoc -w ScatterSim.gamma
pydoc -w ScatterSim.IntegralModel
pydoc -w ScatterSim.MultiComponentModel-AdditionalLattices
pydoc -w ScatterSim.MultiComponentModel
pydoc -w ScatterSim.physcon
pydoc -w ScatterSim.Potentials
pydoc -w ScatterSim.Scattering
pydoc -w ScatterSim.Simulations
pydoc -w ScatterSim.SolutionModel
mv *html doc
