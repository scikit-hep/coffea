# Getting Started

Coffea couples the columnar data model of Awkward Array with a thin execution layer so that an analysis can move from a laptop to a cluster without rewrites.
The workflow always follows the same pattern:

1. Implement a {class}`~coffea.processor.ProcessorABC` that turns event data into accumulators (objects like histograms or sums that summarize the event data in a helpful or interesting way).
2. Execute it with {class}`~coffea.processor.Runner` using a local
   (i.e. only on the computer you are developing on) executor while you iterate.
3. Swap the executor when you are ready to launch the analysis over a larger data set (aka "scale out").

This overall workflow is pretty abstract. The rest of these sections are focused on helping
explain more of the vocabulary used regularly within coffea and its documentation ({doc}`concepts`),
how to install ``coffea`` ({doc}`installation`),
writing your first ``Processor`` ({doc}`develop`),
and scaling out your first ``Processor`` to be able to analyze larger amounts of data ({doc}`scale`).

```{toctree}
:maxdepth: 1
:caption: Getting Started
concepts.md
installation.md
develop.md
scale.md
```
