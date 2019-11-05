// oscilliscope.js
//
// A type of object for modeling an oscilliscope.  Used for displaying 
// fourier polynomials.

// Requiers d3.js

var Oscilloscope = function(fourierp, xs) {

  // Private variables to be initialized at setup time
  var _elem, _width, _height, _svg

  // The fourier polynomial being displayed on the oscilliscope
  var _fourierp = fourierp

  // (xs, ys) is the curve currently displayed on the osciliscope, 
  // it is always some cycling of _fourierp.
  var _xs = (function() {
    if(xs === undefined) {
      return d3.range(0, 1, .0025)
    } else {
      return xs
    }
  })()

  var _ys = _fourierp.eval(_xs)

  return {

    // Bind a the oscilliscope to a dom element
    bind: function(e) {
      var xscale, yscale, data, line
      _elem = e
      _width = parseInt(d3.select(_elem).style("width"))
      //_height = parseInt(d3.select(_elem).style("height"))
      _height = parseInt(50)
      _svg = d3.select(_elem).append("svg")
                             .attr("width", _width)
                             .attr("height", _height)
      return this
    },

    // Render a curve on the osciliscope.  If no path has yet been
    // rendered, construct and render from scratch, otherwise, animate
    // transition to new path, creating the effect that the curve is 
    // translating cyclically.
    render: function() {
      xscale = d3.scale.linear().domain([0, 1]).range([0, _width])
      yscale = d3.scale.linear()
                 .domain([-_fourierp.pmax, _fourierp.pmax])
		             .range([0, _height])
      data = d3.zip(_xs, _ys)
      line = d3.svg.line()
                   .x(function(d) {return xscale(d[0])})
                   .y(function(d) {return yscale(d[1])})
		               .interpolate("linear")
      _svg.append("path").attr("d", line(data))
                         .attr("stroke", "black")
                         .attr("stroke-width", 2)
                         .attr("fill", "none")
    }

  }

}
