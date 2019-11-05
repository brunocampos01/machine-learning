// fourier-polynomial.js
//
// A class modeling a fourier polynomial (finite degree).

var FourierPolynomial = function(cs) {
  
  var _coefs = cs

  return {
  
    get degree() {
      return _coefs.length - 1
    },

    // The maximum possible value obtained by the polynomial
    get pmax() {
      var s, i
      s = 0
      for(i = 0; i < _coefs.length; i++) {
        s += Math.abs(_coefs[i])
      }
      return s
    },

    // Evaluate the polynomial at a number after shifting by a given 
    // phase.
    _eval: function(x, phase) {
      var acc, i
      phase = phase || 0
      acc = 0
      for(i = 0; i <= this.degree; i++) {
        acc += _coefs[i] * Math.sin(2 * i * Math.PI * (x - phase))
      }
      return acc
    },

    // Evaluate teh polynomial at an array after shifting a given 
    // pahse.
    eval: function(xs, phase) {
      that = this
      var ys = xs.map(function(x) {return that._eval(x, phase)})
      that = null // Destruct
      return ys
    },

  }
}

var _runif = function(low, high) {
    return (high - low) * Math.random() + low
}

var randomPolynomial = function(degree) {
    var coefs = []
    for(var i = 0; i <= degree; i++) {
        coefs.push(_runif(-1, 1))
    }
    //coefs = coefs.sort(function(a, b) {return b - a})
    console.log(coefs)
    return new FourierPolynomial(coefs)
}
