#import "@preview/lilaq:0.5.0" as lilaq

#let data = json("results.json")
#show: lilaq.set-legend(position: bottom)

#for (benchmark, details) in data {

  let (units, n, vec_cell_performance, hash_cell_performance, all_pairs_performance) = details 
  
  lilaq.diagram(
    title: [#benchmark],

    xlabel: [$n$],
    ylabel: [#units],
    xscale: "log",
    yscale: "log",

    lilaq.plot(n, vec_cell_performance, label: [`VecCell`]),
    lilaq.plot(n, hash_cell_performance, label: [`HashCell`]),
    lilaq.plot(n, all_pairs_performance, label: [`AllPairs`]),
  )

  h(4mm)
  }
