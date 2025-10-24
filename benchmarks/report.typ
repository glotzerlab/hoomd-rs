#import "@preview/lilaq:0.5.0" as lilaq


#let data = json("results.json")

#for (benchmark, details) in data {

  let (units, n, performance) = details 
  
  lilaq.diagram(
    title: [#benchmark],

    xlabel: [$n$],
    ylabel: [#units],
    xscale: "log",
    yscale: "log",

    lilaq.plot(n, performance),
  )

  h(4mm)
  }
